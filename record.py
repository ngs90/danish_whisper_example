import os
import datetime
import asyncio
import wave
import pyaudio

CHUNK = 1024  # Record in chunks of 1024 samples
SAMPLE_FORMAT = pyaudio.paInt32 #   # 32/16 bits per sample
CHANNELS = 2
FS = 16000 # 44100  # Record at 44100 samples per second
SECONDS = 30

p = pyaudio.PyAudio()  # Create an interface to PortAudio

def get_device_index(): 

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if (('Stereo Mix' in dev['name']) and (dev['hostApi'] == 0)):
            dev_index = dev['index']
            print('device index', dev_index)
    return dev_index 

async def record(fs, seconds, chunk, dev_index):

    print('Recording')
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    stream = p.open(format=SAMPLE_FORMAT,
                    channels=CHANNELS,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                    input_device_index=dev_index)

    now = datetime.datetime.now().strftime(f'%Y-%m-%d-%H-%M-%S-%f')
    filename = f"output_{now}.wav"
    path = os.path.join('recordings', filename)

    frames = []  # Initialize array to store frames

    # Store data in chunks for x seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    return frames, path

async def save(frames, path):
    # Save the recorded data as a WAV file
    if frames is not None:
        print('Saving file')
        wf = wave.open(path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(FS)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f'File {path} was saved!')
    else:
        print('No action yet!')

async def main(dev_index, frames=None, path=None):

    await save(frames, path)
    frames_new, path_new = await record(fs=FS, seconds=SECONDS, chunk=CHUNK, dev_index=dev_index)

    return frames_new, path_new

if __name__ == '__main__':
    dev_index = get_device_index()

    frames, path = None, None
    i = 0

    while True:
        print(i)
        frames, path = asyncio.run(main(dev_index, frames, path))
        i = i + 1
