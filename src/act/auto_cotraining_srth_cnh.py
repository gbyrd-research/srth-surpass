import subprocess
import time

def is_process_running(process_name):
    try:
        # Check for the process using pgrep
        result = subprocess.run(["pgrep", "-f", process_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return len(result.stdout) > 0
    except Exception as e:
        print(f"Error checking process: {e}")
        return False

def run_training():
    # Wait for the yolo_segment process to finish
    # process_name = "python yolo_segment"
    # print(f"Waiting for '{process_name}' process to finish...")
    # while is_process_running(process_name):
    #     time.sleep(5)  # Check every 5 seconds
    
    # print(f"'{process_name}' has finished. Starting training process...")
    
    while True:
        try:
            ## calculate time taken for training
            start = time.time()

            process = subprocess.Popen(
                ["python", "imitate_episodes.py", 
                 "--task_name", "cnh_exvivo",  "exvivo_29_auto_label_srt", 
                 "--ckpt_dir", "ckpt_dir_srth_cnh_cotraining", 
                #  "--dataset_weights" , "5.0 1.0",
                 "--policy_class", "SRT", 
                 "--kl_weight", "10", 
                 "--chunk_size", "60", 
                 "--hidden_dim", "512", 
                 "--batch_size", "16", 
                 "--dim_feedforward", "3200", 
                 "--num_epochs", "20000", 
                 "--lr", "1e-5", 
                 "--seed", "0", 
                 "--use_language", 
                 "--language_encoder", "distilbert", 
                 "--image_encoder", "efficientnet_b3film",
                 "--gpu", "1",
                 "--log_wandb",
                 #"--multi_gpu",
                 "--policy_level", "low"
                ]
            )
            process.wait()

            end = time.time()
            print(f"Time taken: {end-start} seconds")
            if process.returncode == 0:
                print("Training completed successfully.")
                break
            else:

                print(f"Training process interrupted with return code {process.returncode}. Restarting...")
                time.sleep(5)  # Optional: wait before restarting

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

if __name__ == "__main__":
    run_training()
