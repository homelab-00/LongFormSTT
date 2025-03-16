import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:  # Focus on input devices
        print(f"Index {i}: {dev_info['name']} (Sample Rate: {dev_info['defaultSampleRate']} Hz)")