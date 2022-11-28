import urllib
import time
import pathlib
import pandas as pd
import streamlit as st
import os 

TRANSCRIPTIONS_PATH = 'transcriptions' # r'C:\working\ngs-lib\streaming\transcriptions'
if not os.path.exists(TRANSCRIPTIONS_PATH): os.mkdir(TRANSCRIPTIONS_PATH)

class TranscriptionFiles:

    def __init__(self, path):
        self.files = list(pathlib.Path(path).glob('*.txt'))

    def load(self):
        """Load transcription files and return in pandas.DataFrame
        """

        transcriptions = []
        file_paths = []

        for file_path in self.files:
            # print('file_path', file_path)
            text = file_path.open('r').read()
            transcriptions.extend([text])
            file_paths.extend([file_path.name])

        df = pd.DataFrame(dict(text=transcriptions, file_path=file_paths))
        return df 

def get_transcriptions_data():
    transcriptions = TranscriptionFiles(path=TRANSCRIPTIONS_PATH)
    df_text = transcriptions.load()
    return df_text.set_index('file_path')

try:
    placeholder = st.empty()

    while True:

        with placeholder.container():
            df = get_transcriptions_data()
            st.write(" ".join(df.sort_index(ascending=True).tail(50)['text'].to_list()))
            time.sleep(15)


except urllib.error.URLError as e:
    st.error(f"""Error. Reason: {e.reason}""")

# https://discuss.streamlit.io/t/update-dataframe/8798
# mutate chart? https://docs.streamlit.io/library/api-reference/mutate
# https://github.com/ash2shukla/streamlit-stream
