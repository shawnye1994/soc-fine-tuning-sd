from svd_trainers.soc_trainer import SOCTrainer
from video_config_utils import load_config
from diffusers.utils import load_image, export_to_video
import torch
from soc_pipeline_svd import latent_to_decode

if __name__ == '__main__':
    config = load_config("/gpfs-flash/junlab/yexi24-postdoc/soc-fine-tuning-sd/configs/svd_buffer.yaml")
    trainer = SOCTrainer(config).to('cuda:8')
    trainer.on_fit_start()

    trainer.on_save_checkpoint({})

    # inference the svd pipeline to get trajectories
    use_soc_scheduler = True
    use_init_model = False
    learn_offset = False
    store_traj = True
    store_noise = True
    store_noise_pred = True
    
    # pipeline = SOCStableVideoDiffusionPipeline.from_pretrained(
    #             "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.bfloat16
    #             )
    pipeline = trainer.soc_pipeline
    # pipeline.enable_model_cpu_offload()
    if use_soc_scheduler:
        pipeline.set_edm_ancestral_scheduler()
    svd_pipeline = pipeline
    
    target_size = (1024, 576)
    init_frame = load_image("/gpfs-flash/junlab/yexi24-postdoc/TrainingFree3DVideoGeneration/rocket.png")
    init_frame = init_frame.resize(target_size)
    height, width = target_size[1], target_size[0]

    num_inference_steps = 25
    rand_seed = 256
    generator = torch.manual_seed(rand_seed)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = svd_pipeline(init_frame, generator = generator, height=height, width=width, num_frames=24, num_inference_steps=num_inference_steps, fps=7, noise_aug_strength=0.0, 
                                output_type = 'latent',
                                use_soc_scheduler=use_soc_scheduler, use_init_model=use_init_model, learn_offset=learn_offset, 
                                store_traj=store_traj, store_noise=store_noise, store_noise_pred=store_noise_pred
                                )
        # load the soc trainer
        frames, noises, noise_preds, trajectories, image_latents, image_embeddings, added_time_ids, timesteps = out
        # with torch.no_grad():
        #     frames = latent_to_decode(model=svd_pipeline, output_type='pil', latents=out[0], decode_chunk_size=16)
        #     fake_video = svd_pipeline.video_processor.postprocess_video(video=frames, output_type='pil')[0]
        #     # fake_video = out[0]
        #     export_to_video(fake_video, f"./lora_generated_ancestral_{num_inference_steps}_rand{rand_seed}.mp4", fps=7)
        indices_t = [0, 1, 22, 23]
        control_times_sqrt_dt, prev_sample, noise_pred_eval, noise_pred_init_eval, std_dev_t = trainer.evaluate_controls(
            batch_size=frames.shape[0],
            num_timesteps_to_load=len(indices_t),
            image_latents=image_latents,
            image_embeddings=image_embeddings,
            added_time_ids=added_time_ids,
            all_x_t=trajectories,
            all_t=timesteps,
            indices_t=indices_t,
            noise_pred_init=None,
            generator=None
        )

        import pdb; pdb.set_trace()