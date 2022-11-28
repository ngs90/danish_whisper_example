import pathlib
import time
import whisper
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
import os 

AUDIO_PATH = 'recordings' # r'C:\working\ngs-lib\streaming\recordings'
ARCHIVE_PATH = 'archive' # r'C:\working\ngs-lib\streaming\archive'
TRANSCRIPTIONS_PATH = 'transcriptions' # r'C:\working\ngs-lib\streaming\transcriptions'

MODEL_NAME = 'medium' # tiny base small medium large.
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for path in [AUDIO_PATH, ARCHIVE_PATH, TRANSCRIPTIONS_PATH]:
    if not os.path.exists(path):
        os.mkdir(AUDIO_PATH)

class SoundStreamDataset(torch.utils.data.Dataset):
    """Loads .wav files to be consumed by Whisper model

    Args:
        torch (_type_): Torch audio dataset.
    """

    def __init__(self, path, device=DEVICE):
        self.path = path
        self.files = list(pathlib.Path(path).glob('*.wav'))
        self.device = device

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        #signal = whisper.load_audio(audio_path)
        signal, _ = torchaudio.load(str(audio_path))

        # Trim to 30 seconds
        signal = whisper.pad_or_trim(signal.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(signal)

        return (mel, audio_path.name)

    def _get_audio_path(self, index):
        file_path = self.files[index]
        return file_path

if __name__ == '__main__':

    # See all avaiable models here: https://github.com/openai/whisper#available-models-and-languages

    # Initialize model
    model = whisper.load_model(name=MODEL_NAME)
    language = 'da'
    options = whisper.DecodingOptions(language=language, without_timestamps=False, fp16=False)

    print('Model characteristics')
    print(
        f"\t Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )

    while True:
        print('(Re)load the SoundStream Dataset')

        dataset = SoundStreamDataset(path=AUDIO_PATH, device=DEVICE)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

        print('Data characteristics')
        print('\t Number of audiofiles', len(dataset))

        # If no files, then wait for new files to arrive
        if len(dataset) == 0:
            print('No files to process. Waiting for data to arrive.')
            time.sleep(15)
            continue

        print('\t First file:', dataset.files[0])

        hypothesis = []
        audio_paths = []

        for (mels, filenames) in tqdm(loader):
            time_it = time.time()
            print(mels.shape)

            results = model.decode(mels, options)
            print('\t Result of first element in batch:', results[0])

            # Save transcription
            for result, filename in zip(results, filenames):
                transcription_save_path = pathlib.Path(TRANSCRIPTIONS_PATH) \
                                        / filename.replace('.wav', '.txt')
                transcription_save_path.open('w').write(result.text)

                # move file to archive 
                audio_path = pathlib.Path(AUDIO_PATH) \
                                        / filename
                audio_path.rename(pathlib.Path(ARCHIVE_PATH) / filename)

            print('time taken', time.time()-time_it)


            # move files to archive

    # data = pd.DataFrame(dict(hypothesis))
    # print(data)
    ####

    # time_it = time.time()
    # filename = r'recordings/output_2022-11-26-02-06-01-149074.wav'
    # model = whisper.load_model('medium')
    # result = model.transcribe(filename, language='da')

    # print(result['text'])
    # print('time taken', time.time()-time_it)
