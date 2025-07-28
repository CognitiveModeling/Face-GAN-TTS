import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl

from model.face_tts import FaceTTS
from model.feature_extractor import VoiceFeatureExtractor
from model.discriminator import SpectrogramDiscriminator

class FaceTTSWithDiscriminator(FaceTTS):
    def __init__(self, _config):
        super().__init__(_config)
        self.config = _config

        # Instantiate discriminator and feature extractor.
        self.discriminator = SpectrogramDiscriminator(_config)
        self.feature_extractor = VoiceFeatureExtractor(_config)
        self.recon_criterion = nn.L1Loss()

        # ------------------------------------------------------------------ #
        #  Hyperparameter
        # ------------------------------------------------------------------ #
        self.lambda_adv = _config["lambda_adv"]  # small adversarial weight initially
        self.warmup_disc_epochs = _config["warmup_disc_epochs"]  # skip disc updates for these epochs
        self.freeze_gen_epochs = _config["freeze_gen_epochs"] # freeze generator for these epochs
        self.disc_loss_type = _config["disc_loss_type"]
        self.speaker_loss_weight = _config["gamma"]
        self.use_pitch_loss = int(_config.get("use_pitch_loss", 0))
        self.use_energy_loss = int(_config.get("use_energy_loss", 0))
        self.use_fm_loss = int(_config.get("use_fm_loss", 0))


        # ------------------------------------------------------------------ #
        #  Loss Functions
        # ------------------------------------------------------------------ #
        if self.disc_loss_type == "bce":
            self.adv_criterion = nn.BCEWithLogitsLoss()
        elif self.disc_loss_type == "mse":
            self.adv_criterion = nn.MSELoss()
        elif self.disc_loss_type == "hinge": #best so far
            def discriminator_hinge_loss(real_logits, fake_logits):
                loss_real = torch.mean(F.relu(1. - real_logits))
                loss_fake = torch.mean(F.relu(1. + fake_logits))
                return loss_real + loss_fake

            def generator_hinge_loss(fake_logits):
                return -torch.mean(fake_logits)

            self.disc_loss_fn = discriminator_hinge_loss
            self.gen_loss_fn = generator_hinge_loss
        else:
            # Fallback:
            self.adv_criterion = nn.BCEWithLogitsLoss()

        # Disable automatic optimization (update manually)
        self.automatic_optimization = False

    # ------------------------------------------------------------------ #
    #  Loss Help Functions
    # ------------------------------------------------------------------ #
    def compute_pitch_loss(self, real_f0, fake_f0):
        if real_f0.size(-1) != fake_f0.size(-1): # padding if input size not the same
            if real_f0.size(-1) > fake_f0.size(-1):
                real_f0 = real_f0[..., :fake_f0.size(-1)]
            else:
                fake_f0 = F.pad(fake_f0, (0, real_f0.size(-1) - fake_f0.size(-1)))

        voiced_mask = (real_f0 > 1e-5) & (fake_f0 > 1e-5)
        if voiced_mask.any():
            return self.recon_criterion(real_f0[voiced_mask], fake_f0[voiced_mask])
        else:
            return torch.tensor(0.0, device=real_f0.device)

    def compute_energy_loss(self, real_energy, fake_energy):
        if real_energy.size(-1) != fake_energy.size(-1):
            if real_energy.size(-1) > fake_energy.size(-1):
                real_energy = real_energy[..., :fake_energy.size(-1)]
            else:
                fake_energy = F.pad(fake_energy, (0, real_energy.size(-1) - fake_energy.size(-1)))
        return self.recon_criterion(real_energy, fake_energy)

    def compute_feature_matching_loss(self, real_features, fake_features):
        fm_loss = 0.0
        for real, fake in zip(real_features, fake_features):
            if real.numel() == 0 or fake.numel() == 0:
                continue

            min_len = min(real.size(-1), fake.size(-1))
            real, fake = real[..., :min_len], fake[..., :min_len]
            fm_loss += F.l1_loss(real, fake)
        return fm_loss

    # ---------------------------------------------------------------------- #
    #  Lightning-Lifecycle
    # ---------------------------------------------------------------------- #
    def on_train_start(self):
        # Optionally freeze generator components at the start
        if self.freeze_gen_epochs > 0:
            print(f"[INFO] Freezing generator for the first {self.freeze_gen_epochs} epochs.")
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        # Unfreeze generator after freeze_gen_epochs
        if self.freeze_gen_epochs > 0 and self.current_epoch >= self.freeze_gen_epochs:
            print("[INFO] Unfreezing generator now.")
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.decoder.parameters():
                p.requires_grad = True
            self.freeze_gen_epochs = 0  # Do this only once

    def configure_optimizers(self):
        # Optionally use separate learning rates
        gen_lr = self.config["learning_rate"]
        disc_lr = self.config.get("disc_learning_rate", gen_lr)
        generator_optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=gen_lr
        )
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=disc_lr)
        return [generator_optimizer, discriminator_optimizer]

    def training_step(self, batch, batch_idx):
        # Move data to device
        x = batch['x'].to(self.device)
        x_len = batch['x_len']
        y = batch['y'].to(self.device)
        y_len = batch['y_len']
        spk = batch['spk'].to(self.device)
        B = x.shape[0]

        # Get minibatch sizes (defined unconditionally)
        micro_batch_size = self.config.get("micro_batch_size", 16)
        micro_batch_size_gen = self.config.get("micro_batch_size_gen", micro_batch_size)
        n_micro_batches = (B + micro_batch_size - 1) // micro_batch_size
        n_micro_batches_gen = (B + micro_batch_size_gen - 1) // micro_batch_size_gen

        # Get optimizers.
        opt_gen, opt_disc = self.optimizers()

        # Determine whether to update the discriminator based on a warm-up
        train_disc = self.current_epoch >= self.warmup_disc_epochs

        # -------- Discriminator-Step ------------------------------------
        if train_disc:
            opt_disc.zero_grad()
            total_d_loss = 0.0
            for i in range(n_micro_batches):
                # ---------------- Data in Mini-Batches ----------------
                start = i * micro_batch_size
                end = min((i + 1) * micro_batch_size, B)
                x_mini = x[start:end]
                x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
                y_mini = y[start:end]
                y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
                spk_mini = spk[start:end]

                # Generate fake mel-spectrograms (without gradient update)
                with torch.no_grad():
                    _, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
                fake_mel = dec_out[-1]

                # Run discriminator on real and fake.
                _, real_logits = self.discriminator(y_mini.unsqueeze(1))
                _, fake_logits = self.discriminator(fake_mel.detach().unsqueeze(1))

                if self.disc_loss_type == "hinge":
                    d_loss = self.disc_loss_fn(real_logits, fake_logits)
                else:
                    loss_real = self.adv_criterion(real_logits, torch.ones_like(real_logits))
                    loss_fake = self.adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (loss_real + loss_fake)

                # Accuracy
                with torch.no_grad():
                    if self.disc_loss_type == "hinge":
                        real_acc = (real_logits > 0).float().mean()
                        fake_acc = (fake_logits < 0).float().mean()
                    else:
                        real_acc = ((torch.sigmoid(real_logits) > 0.5).float()).mean()
                        fake_acc = ((torch.sigmoid(fake_logits) < 0.5).float()).mean()
                    disc_acc = 0.5 * (real_acc + fake_acc)
                    self.log("train/disc_acc", disc_acc, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

                # -- R1 regularization on real samples --
                # -- Optional R1 regularization on real samples --
                if self.config.get("use_r1_penalty", 1) and self.current_epoch >= self.config.get("r1_start_epoch", 0):
                    r1_gamma = self.config.get("r1_gamma", 10.0)
                    real_data = y_mini.unsqueeze(1).detach()
                    real_data.requires_grad_(True)
                    _, real_logits_r1 = self.discriminator(real_data)
                    r1_loss = real_logits_r1.sum()
                    grad_real = torch.autograd.grad(outputs=r1_loss,
                                                    inputs=real_data,
                                                    create_graph=True)[0]
                    r1_penalty = grad_real.pow(2).sum(dim=[1,2,3]).mean()
                    d_loss = d_loss + r1_gamma * 0.5 * r1_penalty

                if torch.isnan(d_loss) or torch.isinf(d_loss):
                    print(f"[WARNING] NaN or Inf detected in d_loss at step {batch_idx}. Skipping update.")
                    continue

                self.manual_backward(d_loss / n_micro_batches)
                total_d_loss += d_loss.item()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1)
            opt_disc.step()
            avg_d_loss = total_d_loss / n_micro_batches if train_disc else 0.0

        else:
            avg_d_loss = 0.0

        self.log("train/d_loss", avg_d_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # -------- Generator-Step ---------------------------------------
        opt_gen.zero_grad()

        total_vals = {
            "g"      : 0.0,
            "adv"    : 0.0,
            "dur"    : 0.0,
            "prior"  : 0.0,
            "diff"   : 0.0,
            "spk"    : 0.0,
            "pitch"  : 0.0,
            "energy" : 0.0,
            "fm"     : 0.0,
        }

        for i in range(n_micro_batches_gen):
            # ---------------- Data in Mini-Batches ----------------
            start = i * micro_batch_size_gen
            end = min((i + 1) * micro_batch_size_gen, B)
            x_mini = x[start:end]
            x_len_mini = x_len[start:end] if hasattr(x_len, '__getitem__') else x_len
            y_mini = y[start:end]
            y_len_mini = y_len[start:end] if hasattr(y_len, '__getitem__') else y_len
            spk_mini = spk[start:end]

            enc_out, dec_out, _ = self.forward(x_mini, x_len_mini, self.config['timesteps'], spk=spk_mini)
            fake_mel = dec_out[-1]

            # ---------------- If discriminator is active, compute adversarial loss ----------------
            if train_disc:
                fake_fmap, fake_logits = self.discriminator(fake_mel.unsqueeze(1))
                if self.disc_loss_type == "hinge":
                    adv_loss = self.gen_loss_fn(fake_logits)
                else:
                    adv_loss = self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
            else:
                fake_fmap = None
                adv_loss = torch.zeros(1, device=self.device, requires_grad=True)
            
            # feature matching
            if self.use_fm_loss and train_disc:
                with torch.no_grad():
                    real_fmap, _ = self.discriminator(y_mini.unsqueeze(1))
                fm_loss = self.compute_feature_matching_loss(real_fmap, fake_fmap)
            else:
                fm_loss = torch.tensor(0.0, device=self.device)

            # ---------------- optional pitch / energy ----------------
            with torch.no_grad():
                real_cpu = y_mini[0].detach().cpu().numpy()
                fake_cpu = fake_mel[0].detach().cpu().numpy()

            if self.use_pitch_loss:
                real_f0  = self.feature_extractor.extract_f0(real_cpu)
                fake_f0  = self.feature_extractor.extract_f0(fake_cpu)
                pitch_loss = self.compute_pitch_loss(real_f0, fake_f0).to(self.device)
            else:
                pitch_loss = torch.tensor(0.0, device=self.device)

            if self.use_energy_loss:
                real_e = self.feature_extractor.extract_energy(real_cpu)
                fake_e = self.feature_extractor.extract_energy(fake_cpu)
                energy_loss = self.compute_energy_loss(real_e, fake_e).to(self.device)
            else:
                energy_loss = torch.tensor(0.0, device=self.device)

            # ---------------- Compute the original FaceTTS losses ----------------
            dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(
                x_mini, x_len_mini, y_mini, y_len_mini, spk=spk_mini
            )
            spk_loss_weighted = self.speaker_loss_weight * spk_loss

            g_loss = (
                self.lambda_adv * adv_loss +
                dur_loss + prior_loss + diff_loss +
                spk_loss_weighted +
                self.use_fm_loss * fm_loss +
                self.use_pitch_loss * pitch_loss +
                self.use_energy_loss * energy_loss
            )

            self.manual_backward(g_loss / n_micro_batches_gen)
            
            total_vals["g"]      += g_loss.item()
            total_vals["adv"]    += adv_loss.item()
            total_vals["dur"]    += dur_loss.item()
            total_vals["prior"]  += prior_loss.item()
            total_vals["diff"]   += diff_loss.item()
            total_vals["spk"]    += spk_loss.item()
            total_vals["pitch"]  += pitch_loss.item()
            total_vals["energy"] += energy_loss.item()
            total_vals["fm"]     += fm_loss.item()


        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1)
        opt_gen.step()

        if self.global_step % 500 == 0:       
            torch.cuda.empty_cache()

        # ---------------- Mean per global step -------------------
        inv = 1.0 / n_micro_batches_gen
        log_vals = {
            "train/g_loss"        : total_vals["g"]    * inv,
            "train/adv_loss"      : total_vals["adv"]  * inv,
            "train/duration_loss" : total_vals["dur"]  * inv,
            "train/prior_loss"    : total_vals["prior"]* inv,
            "train/diffusion_loss": total_vals["diff"] * inv,
            "train/spk_loss"      : total_vals["spk"]  * inv,
        }
        if self.use_pitch_loss:
            log_vals["train/pitch_loss"] = total_vals["pitch"] * inv
        if self.use_energy_loss:
            log_vals["train/energy_loss"] = total_vals["energy"] * inv
        if self.use_fm_loss:
            log_vals["train/fm_loss"] = total_vals["fm"] * inv

        self.log_dict(log_vals, on_step=True, on_epoch=True, sync_dist=True)

        # ---------------- Mean Epoch ----------------------------
        if (batch_idx + 1) == len(self.trainer.datamodule.train_dataloader()):
            self.log("train/g_loss_epoch", log_vals["train/g_loss"],
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/d_loss_epoch", avg_d_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Meta-Info
        if self.global_step == 0 and batch_idx == 0:
            self.log("hp/micro_bs", micro_batch_size, prog_bar=False)

        return {"d_loss": avg_d_loss, "g_loss": log_vals["train/g_loss"]}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x, x_len, y, y_len, spk = batch["x"], batch["x_len"], batch["y"], batch["y_len"], batch["spk"]

        # ---------- Generator forward pass ----------
        _, dec_out, _ = self.forward(x, x_len, self.config["timesteps"], spk=spk)
        fake_mel = dec_out[-1]

        # ---------- Adversarial & FM loss ----------
        if self.current_epoch >= self.warmup_disc_epochs:
            fake_fmap, fake_logits = self.discriminator(fake_mel.unsqueeze(1))
            adv_loss = (
                self.gen_loss_fn(fake_logits)
                if self.disc_loss_type == "hinge"
                else self.adv_criterion(fake_logits, torch.ones_like(fake_logits))
            )
            fm_loss = torch.tensor(0.0, device=self.device)
            if self.use_fm_loss:
                real_fmap, _ = self.discriminator(y.unsqueeze(1))
                fm_loss = self.compute_feature_matching_loss(real_fmap, fake_fmap)
        else:
            adv_loss = torch.zeros(1, device=self.device)
            fm_loss = torch.zeros(1, device=self.device)

        # ---------- Optional pitch / energy ----------
        pitch_loss, energy_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        if self.use_pitch_loss:
            real_f0  = self.feature_extractor.extract_f0(y[0].cpu().numpy()).to(self.device)
            fake_f0  = self.feature_extractor.extract_f0(fake_mel[0].cpu().numpy()).to(self.device)
            pitch_loss = self.compute_pitch_loss(real_f0, fake_f0)
        if self.use_energy_loss:
            real_e = self.feature_extractor.extract_energy(y[0].cpu().numpy()).to(self.device)
            fake_e = self.feature_extractor.extract_energy(fake_mel[0].cpu().numpy()).to(self.device)
            energy_loss = self.compute_energy_loss(real_e, fake_e)

        # ---------- Core FaceTTS losses ----------
        dur_loss, prior_loss, diff_loss, spk_loss = self.compute_loss(
            x, x_len, y, y_len, spk=spk, out_size=self.config["out_size"]
        )
        spk_loss_weighted = self.speaker_loss_weight * spk_loss

        # ---------- Total validation loss ----------
        val_loss = (
            self.lambda_adv * adv_loss
            + dur_loss + prior_loss + diff_loss
            + spk_loss_weighted
            + self.use_fm_loss * fm_loss
            + self.use_pitch_loss * pitch_loss
            + self.use_energy_loss * energy_loss
        )

        # ---------- Logging ----------
        log_dict = {
            "val/adv_loss": adv_loss,
            "val/duration_loss": dur_loss,
            "val/prior_loss": prior_loss,
            "val/diffusion_loss": diff_loss,
            "val/spk_loss": spk_loss,
            "val/total_loss": val_loss,
        }
        if self.use_fm_loss:     log_dict["val/fm_loss"]     = fm_loss
        if self.use_pitch_loss:  log_dict["val/pitch_loss"]  = pitch_loss
        if self.use_energy_loss: log_dict["val/energy_loss"] = energy_loss

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return val_loss                                           
