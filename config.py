import os
from sacred import Experiment

from model.utils import fix_len_compatibility

ex = Experiment("face-tts")


@ex.config
def config():

    seed = int(os.getenv("seed", 37))
    perceptual_loss = int(os.getenv("perceptual_loss", 1))  # True: 1 / False: 0 / generates xhat for Speaker Feature Binding Loss / Has to be true otherwise error /
    #local_checkpoint_dir = os.getenv("local_checkpoint_dir", "./checkpoints")

    # Dataset Configs
    dataset = os.getenv("dataset", "lrs2")
    lrs2_train = os.getenv("lrs2_train", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_train_long.list")
    lrs2_val = os.getenv("lrs2_val", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_val_long.list")
    lrs2_test = os.getenv("lrs2_test", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_test_long.list")
    lrs2_path = os.getenv("lrs2_path", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted")
    cmudict_path = os.getenv("cmudict_path", "utils/cmu_dictionary")

    # Data Configs
    image_size = int(os.getenv("image_size", 224))
    max_frames = int(os.getenv("max_frames", 30))
    image_augment = int(os.getenv("image_augment", 0))

    ## hifigan-16k setting
    n_fft = int(os.getenv("n_fft", 1024))
    sample_rate = int(os.getenv("sample_rate", 16000))
    hop_len = int(os.getenv("hop_len", 160))
    win_len = int(os.getenv("win_len", 1024))
    f_min = float(os.getenv("f_min", 0.0))
    f_max = float(os.getenv("f_max", 8000))
    n_mels = int(os.getenv("n_mels", 128))
    # Network Configs

    ## Encoder parameters
    n_feats = n_mels
    spk_emb_dim = int(os.getenv("spk_emb_dim", 64))  # For multispeaker Grad-TTS
    vid_emb_dim = int(os.getenv("vid_emb_dim", 512))  # For Face-TTS
    n_enc_channels = int(os.getenv("n_enc_channels", 192))
    filter_channels = int(os.getenv("filter_channels", 768))
    filter_channels_dp = int(os.getenv("filter_channels_dp", 256))
    n_enc_layers = int(os.getenv("n_enc_layers", 6))
    enc_kernel = int(os.getenv("enc_kernel", 3))
    enc_dropout = float(os.getenv("enc_dropout", 0.0))
    n_heads = int(os.getenv("n_heads", 2))
    window_size = int(os.getenv("window_size", 4))

    ## Decoder parameters
    dec_dim = int(os.getenv("dec_dim", 64))
    beta_min = float(os.getenv("beta_min", 0.05))
    beta_max = float(os.getenv("beta_max", 20.0))
    pe_scale = float(os.getenv("pe_scale", 1000.0))

    ## Syncnet parameters
    syncnet_stride = int(os.getenv("syncnet_stride", 1))
    syncnet_ckpt = os.getenv("syncnet_ckpt")
    spk_emb = os.getenv("spk_emb", "face") #or "speech"

    # Experiment Configs for original facetts- already best combination choosen
    batch_size = int(os.getenv("batch_size", 256)) #it was 256 #if gpu =4 -> 256%4 // For batch_size=64 no cuda out memory for gan
    add_blank = int(os.getenv("add_blank", 1))  # True: 1 / False: 0
    snet_emb = int(os.getenv("snet_emb", 1))  # True: 1 / False: 0
    n_spks = int(os.getenv("n_spks", 7358))  # libritts:247, lrs3: 2007, lrs2: 7358
    multi_spks = int(os.getenv("multi_spks", 1))
    out_size = fix_len_compatibility(2 * sample_rate // 256)
    #model = os.getenv("model", "face-tts")
    
    denoise_factor = float(os.getenv("denoise_factor", 0.7))

    #ONLY FOR TESTING THE PREPROCESSING PIPELINE
    use_bandstop_filter = int(os.getenv("use_bandstop_filter", 0))  # 0 = off, 1 = on #seemed to be metallic
    # bandstop_center_freq = float(os.getenv("bandstop_center_freq", 202.73)) 
    bandstop_q_value     = float(os.getenv("bandstop_q_value", 1))  

    use_highpass_filter = int(os.getenv("use_highpass_filter", 0))
    highpass_cutoff = float(os.getenv("highpass_cutoff", 70.0)) 

    use_lowpass_filter = int(os.getenv("use_lowpass_filter", 0))       # 0 = deaktiviert, 1 = aktiviert
    lowpass_cutoff = float(os.getenv("lowpass_cutoff", 4500.0))   

    # -----------------------------------------------------------------------------
    # GAN 
    # -----------------------------------------------------------------------------
    use_gan = int(os.getenv("use_gan", 1))  # 0 = False, 1 = True
    use_pitch_loss = int(os.getenv("use_pitch_loss", 0))
    use_energy_loss = int(os.getenv("use_energy_loss", 0))
    use_fm_loss = int(os.getenv("use_fm_loss", 0))
    disc_loss_type = os.getenv("disc_loss_type", "hinge" )  # oder "mse", "hinge" , "bce", "ls" #default hinge 
    lambda_adv = float(os.getenv("lambda_adv",  0.7)) #lambda_adv 0.2 

    gamma = float(os.getenv("gamma", 0.02)) # speaker loss weight for facetts #default for original FaceTTs 0.01 for FaceGANtts 0.02

    #Discriminator Parameter
    disc_lrelu_slope = float(os.getenv("disc_lrelu_slope", 0.3)) 
    disc_learning_rate = float(os.getenv("disc_learning_rate", 1e-4)) 
    use_spectral_norm = int(os.getenv("use_spectral_norm", 0)) # True: 1 / False: 0 

    disc_base_channels = int(os.getenv("disc_base_channels", 64)) 
    disc_num_layers = int(os.getenv("disc_num_layers", 5)) 
    residual_channels = int(os.getenv("residual_channels", 256))
    kernel_width = int(os.getenv("kernel_width", 5)) 
    kernel_height = int(os.getenv("kernel_height", 12)) 
    disc_stride = int(os.getenv("disc_stride", 1))  
    disc_padding = int(os.getenv("disc_padding", 6)) 

    warmup_disc_epochs = int(os.getenv("warmup_disc_epochs", 0)) 
    freeze_gen_epochs = int(os.getenv("freeze_gen_epochs", 0)) 
    micro_batch_size = int(os.getenv("micro_batch_size", 16))

    use_r1_penalty = int(os.getenv("use_r1_penalty", 1))  # 1 = True (default), 0 = False
    r1_gamma = float(os.getenv("r1_gamma", 15.0))         
    r1_start_epoch = int(os.getenv("r1_start_epoch", 0))  


    # Opimizer Configs for Discriminator 
    disc_betas_0 = float(os.getenv("disc_betas_0", 0.9))  # First beta parameter for discriminator Adam # default 0.9 
    disc_betas_1 = float(os.getenv("disc_betas_1", 0.999))  # Second beta parameter for discriminator Adam default 0.999 
    # Discriminator-Optimizer Einstellungen
    disc_eps = float(os.getenv("disc_eps", 1e-8))  
   
    # Optimizer Configs for Generator 
    optim_type = os.getenv("optim_type", "adam") 
    schedule_type = os.getenv("schedule_type", "constant")
    learning_rate = float(os.getenv("learning_rate", 1e-8)) # FaceTTs original no checkpoints default 1e-4 and with 1e-6 FaceGANtts default 1e-8
    end_lr = float(os.getenv("end_lr", 1e-7)) 
    weight_decay = float(os.getenv("weight_decay", 0.1))
    decay_power = float(os.getenv("decay_power", 1.0))
    max_steps = int(os.getenv("max_steps", 100000))

    save_step = int(os.getenv("save_step", 10000))
    warmup_steps = float(os.getenv("warmup_steps", 2))  
    gen_eps = float(os.getenv("gen_eps", 1e-8))  

    video_data_root = os.getenv("video_data_root", "mp4")
    image_data_root = os.getenv("image_data_root", "jpg")
    audio_data_root = os.getenv("audio_data_root", "wav")
    #log_dir = os.getenv("CHECKPOINTS", "./logs")
    log_every_n_steps = int(os.getenv("log_every_n_steps", 1000))

    num_gpus = int(os.getenv("num_gpus", 4)) 
    per_gpu_batchsize = int(batch_size / num_gpus)
    num_nodes = int(os.getenv("num_nodes", 1)) 
    num_workers = int(os.getenv("num_workers",  8))  
    prefetch_factor = int(os.getenv("preftch_factor", 2))  

    #Checkpoints loading
    resume_from = os.getenv("resume_from", "./ckpts/facetts_lrs3.pt") #to train initializing with LRS3 Checkpoints set: ./ckpts/facetts_lrs3.pt # to train from scratch set ./ckpts/no
    # Inference Configs
    test_txt = os.getenv("test_txt", "test/text.txt")
    use_custom = int(os.getenv("use_custom", 2))  # Controls inference mode:
    # 1 = use a custom face image and text,
    # 2 = use a face from the dataset and perform batch inference over LRS2 test set,
    # 0 or other = load a dataset face image but do not run inference

    test_faceimg = os.getenv("test_faceimg", "test/face.png") 
    timesteps = int(os.getenv("timesteps", 10))
    output_dir_orig = os.getenv("output_dir", "/mnt/lustre/work/butz/bst080/faceGANtts/test/CFD_Facetts_scratch")
    output_dir_gan = os.getenv("output_dir", "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_faceGANtts")
    ground_truth_dir =os.getenv("ground_truth_dir", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test/")
    results_path = os.getenv("results_path", "evaluation")
    # SyncNet Configs
    syncnet_initw = float(os.getenv("syncnet_initw", 10.0))
    syncnet_initb = float(os.getenv("syncnet_initb", -5.0))

    #resume checkpoints from for inference
    infr_resume_from_orig = os.getenv("infr_resume_from_orig", "/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561276/checkpoints/epoch=096.ckpt")
    infr_resume_from_gan = os.getenv("infr_resume_from_gan", "/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538393/checkpoints/best_epoch_96_step_17848.ckpt") 

    val_check_interval = float(os.getenv("val_check_interval", 1.0))
    test_only = int(os.getenv("test_only", 0))
    eval_interval = int(os.getenv("eval_interval", 1000)) #training steps interval to evaluate
    early_stopping_patience = int(os.getenv("early_stopping_patience", 30))
    early_stopping_min_delta = float(os.getenv("early_stopping_min_delta", 0.001))

    id = os.getenv("id", "unknown")
    working_dir = os.getenv("working_dir", "")


