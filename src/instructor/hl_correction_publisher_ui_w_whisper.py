import argparse
import os
import sys
from sys import platform
from difflib import SequenceMatcher
import rospy
from std_msgs.msg import String
import time
from pynput import keyboard
from threading import Lock
import signal

# Import the necessary modules from this package
PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
if PATH_TO_YAY_ROBOT:
    sys.path.append(os.path.join(PATH_TO_YAY_ROBOT, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")


# List the phase instructions
phase_instructions = [
    'grabbing gallbladder', 
    'clipping first clip left tube', 
    'going back first clip left tube', 
    'clipping second clip left tube', 
    'going back second clip left tube', 
    'clipping third clip left tube', 
    'going back third clip left tube', 
    'go to the cutting position left tube', 
    'go back from the cut left tube', 
    'clipping first clip right tube', 
    'going back first clip right tube', 
    'clipping second clip right tube', 
    'going back second clip right tube', 
    'clipping third clip right tube', 
    'going back third clip right tube', 
    'go to the cutting position right tube', 
    'go back from the cut right tube'
]

# ROS topics to publish the correction to
instruction_topic_name = '/hl_policy_correction_phase_instruction'


# Signal handler for clean exit
def signal_handler(sig, frame):
    print("\n\nProgram will stop now. Exiting...")
    sys.exit(0)


# Function to display phase options
def display_phases():
    print("Please select a phase instruction:")
    for i, phase in enumerate(phase_instructions, start=1):
        print(f"{i}: {phase}")


# Function to publish the selected phase instruction
def publish_correction_phase_instruction(pub, phase_name):
    rospy.loginfo(f"Publishing correction phase instruction: {phase_name}")
    pub.publish(phase_name)


# Function to find the closest matching phase instruction (Whisper)
def get_closest_phase_instruction(command):
    best_match = None
    highest_ratio = 0.0
    for phase in phase_instructions:
        ratio = SequenceMatcher(None, command.lower(), phase.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = phase
    return best_match, highest_ratio


def main(args):
    rospy.init_node('hl_policy_phase_correction', anonymous=True)
    instruction_pub = rospy.Publisher(instruction_topic_name, String, queue_size=1)
    signal.signal(signal.SIGINT, signal_handler)

    if args.use_audio:
        # Load / Download Whisper model and setup audio
        from aloha_pro.aloha_scripts.audio_utils import AudioTranscriber
        audio_transcriber = AudioTranscriber()
        audio_transcriber.setup_audio(
            model_name=args.model,
            energy_threshold=args.energy_threshold,
            sample_rate=48000 if "linux" in platform else 16000,
            default_microphone=args.default_microphone if "linux" in platform else None,
        )
        print("Audio setup completed.\n")

        recording = False

        def on_press(key):
            nonlocal recording
            if key.char == "2":
                recording = True

        def on_release(key):
            nonlocal recording
            if key.char == "2":
                recording = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while not rospy.is_shutdown():
            display_phases()

            if recording:
                print("Recording command...")
                command = audio_transcriber.transcribe_command(args.record_timeout, args.phrase_timeout)
                if command:
                    best_match, similarity = get_closest_phase_instruction(command)
                    if similarity > args.required_similarity_threshold:  # You can adjust this threshold
                        publish_correction_phase_instruction(instruction_pub, best_match)
                        print(f"Published phase instruction: {best_match}")
                        print(f"Recorded command: {command}, matched with phase: {best_match} to a similarity of {similarity}") 
                    else:
                        print(f"Recorded phase instruction '{command}' is not supported and not similar enough to the given 17 phases. Closest match: {best_match} with similarity {similarity}")
                audio_transcriber.recording = False
                recording = False
            else:
                time.sleep(0.1)
    else:
        while not rospy.is_shutdown():
            display_phases()
            phase_number = int(input("\nEnter the phase number (1-17): "))

            if 1 <= phase_number <= 17:
                phase_name = phase_instructions[phase_number - 1]
                publish_correction_phase_instruction(instruction_pub, phase_name)
            else:
                print("Invalid input. Please enter a number between 1 and 17.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choosing between keyboard input or Whisper audio input
    parser.add_argument(
        "--use_audio",
        action="store_true",
        help="Use Whisper for audio transcription instead of manual input.",
    )
    
    # Whisper arguments
    parser.add_argument(
        "--required_similarity_threshold",
        default=0.6,
        help="Minimum similarity required to match a phase instruction.",
        type=float,
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Model to use for Whisper",
        choices=["tiny", "base", "small", "medium", "large", "large-v2"],
    )
    parser.add_argument(
        "--energy_threshold",
        default=100,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=0.1,
        help="How real-time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=0.5,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="USB PnP Audio Device: Audio",
            help="Default microphone name for SpeechRecognition.",
            type=str,
        )
    main(parser.parse_args())
