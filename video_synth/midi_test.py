# Import the mido library for MIDI communication
import mido
import time

def listen_to_midi_device():
    """
    Connects to the first available MIDI input device and prints incoming messages.
    """
    input_ports = mido.get_input_names()

    if not input_ports:
        print("No MIDI input devices found.")
        print("Please ensure your MIDI device is connected and recognized by your system.")
        return

    print("Available MIDI input devices:")
    for i, port_name in enumerate(input_ports):
        print(f"  {i}: {port_name}")

    # Attempt to connect to the first available input port
    try:
        # You can modify this to let the user select a port, e.g.,
        # selected_index = int(input("Enter the number of the device to listen to: "))
        # port_name_to_open = input_ports[selected_index]
        
        port_name_to_open = input_ports[0] # Automatically pick the first one
        
        print(f"\nAttempting to open MIDI input port: '{port_name_to_open}'")
        with mido.open_input(port_name_to_open) as inport:
            print(f"Successfully opened '{inport.name}'. Listening for MIDI messages...")
            print("Press Ctrl+C to stop listening.")

            # Loop indefinitely to listen for messages
            for msg in inport:
                print(f"Received MIDI message: {msg}")

    except IndexError:
        print("Invalid device selection. Please run the script again and choose a valid number.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nStopped listening to MIDI messages.")

if __name__ == "__main__":
    listen_to_midi_device()
