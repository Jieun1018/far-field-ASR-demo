import gradio as gr
from PIL import Image
import grpc

import protos.asr_pb2 as asr_pb2
import protos.asr_pb2_grpc as asr_pb2_grpc

import os
import base64

import torchaudio

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

keywords = ["북한", "타격", "전쟁", "도발", "공습", "폭음", "총성", "지뢰", "폭탄", "테러"]

# Choices
languages = []
lang_to_code = {}
for code, lang in LANGUAGES.items():
    languages.append(lang)
    lang_to_code[lang] = code
model_choices = os.listdir("models")

# Title
with open("images/hyu_logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()
image_md = f"""
<div align="center">
    <img src="data:image/png;base64,{encoded_string}" alt="ASML Logo" style="max-width:30%;">
</div>
"""

#@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');
# 'Brush Script MT', cursive
text_md = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hahmlet:wght@700&family=Noto+Sans+KR:wght@700&display=swap');
</style>

<div style="text-align: center; font-size: 60px; font-weight: bold; font-family: 'Hahmlet', system-ui;">
Hanyang University ASML ASR
</div>
"""

#color: rgba(243, 176, 195, 1.0);
#color: rgba(254, 121, 213, 0.6);
css = """
.btn-pink {
    background-color: rgba(254, 225, 232, 0.6);
    color: rgba(243, 176, 195, 1.0);
}

.btn-grey {
    background-color: rgba(245, 247, 250, 1.0);
    color: grey;
}
"""

def clear_interface():
    return None, "", None

#def clear_interface():
#    return None, ""

def get_audio_duration_torchaudio(filepath):
    audio_tensor, sample_rate = torchaudio.load(filepath)
    duration = audio_tensor.shape[1] / sample_rate
    return duration

def request(filepath, language, with_timestamps):
    channel = grpc.insecure_channel('localhost:50051')
    stub = asr_pb2_grpc.ASRServiceStub(channel)
    request = asr_pb2.ASRRequest()
    request.filepath = filepath
    request.language = lang_to_code[language]
    request.with_timestamps = with_timestamps
    request.duration = get_audio_duration_torchaudio(filepath)

    # TODO: Asynchronous call
    response = stub.Transcribe(request)

    return response.transcription

def search_kws(transcribed_text):
    detected_keywords = []
    for keyword in transcribed_text.split():
        flag = True
        for i in range(len(keyword)):
            sub_word = keyword[:i+1]
            if sub_word in keywords:
                detected_keywords.append((sub_word, "Detected keyword"))
                detected_keywords.append((keyword[i+1:], "Origin"))
                flag = False
                break
        if flag:
            if keyword in keywords:
                detected_keywords.append((sub_word, "Detected keyword"))
            else:
                detected_keywords.append((keyword, "Origin"))

    return detected_keywords

def reload_model(model_name):
    channel = grpc.insecure_channel('localhost:50051')
    stub = asr_pb2_grpc.ASRServiceStub(channel)
    request = asr_pb2.ReloadModelRequest(model_name=model_name)
    response = stub.ReloadModel(request)
    if response.success:
        gr.Info("Model reloaded")
    else:
        gr.Warning("Model reloading failed")

def create_interface_for_microphone():
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(label='Model', choices=model_choices, value="prompt_nat_v3")
                    reload_button = gr.Button(value="Reload model", icon="images/reload.png")
                    reload_button.click(fn=reload_model, inputs=[model], outputs=[])
                audio_input = gr.Audio(sources="microphone", label='Input Speech', type="filepath")
                language = gr.Dropdown(label='Language', choices=languages, value="korean")
            with gr.Column():
                with_timetstamps = gr.Checkbox(label="With Timestamps", value=False)
                transcribed_text = gr.Textbox(label='Transcribed Text')
                #clear_button = gr.Button('Clear')
                #clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text])
        with gr.Row():
            with gr.Column():
                transcribe_button = gr.Button('Transcribe', elem_classes="btn-pink")
                transcribe_button.click(fn=request, inputs=[audio_input, language, with_timetstamps], outputs=transcribed_text)
                clear_button = gr.Button('Clear', elem_classes="btn-grey")
                clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text])

def create_interface_for_file_upload():
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown(label='Model', choices=model_choices, value="prompt_nat_v3")
                    reload_button = gr.Button(value="Reload model", icon="images/reload.png")
                    reload_button.click(fn=reload_model, inputs=[model], outputs=[])
                audio_input = gr.Audio(sources="upload", label='Input Speech', type="filepath")
                language = gr.Dropdown(label='Language', choices=languages, value="korean")
            with gr.Column():
                with_timetstamps = gr.Checkbox(label="With Timestamps", value=False)
                transcribed_text = gr.Textbox(label='Transcribed Text')
                with gr.Row():
                    predicted_keyword_text = gr.HighlightedText(
                        label='highlight predicted keyword in transcribtion',
                        show_legend=True,
                        color_map={"Detected keyword": "red", "Origin": "green"})
                with gr.Row():
                    pred_kws_button = gr.Button('Predict keyword', elem_classes="btn-orange")
                    pred_kws_button.click(fn=search_kws, inputs=[transcribed_text], outputs=predicted_keyword_text)
                
        with gr.Row():
            with gr.Column():
                transcribe_button = gr.Button('Transcribe', elem_classes="btn-pink")
                transcribe_button.click(fn=request, inputs=[audio_input, language, with_timetstamps], outputs=transcribed_text)
                clear_button = gr.Button('Clear', elem_classes="btn-grey")
                clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text, predicted_keyword_text])
                #clear_button.click(fn=clear_interface, inputs=[], outputs=[audio_input, transcribed_text])

with gr.Blocks(css=css, theme='freddyaboulton/test-blue') as demo:
    gr.Markdown(text_md)
    gr.Markdown(" ")
    #gr.Markdown(image_md)
    with gr.Tab("Transcribe from audio file"):
        create_interface_for_file_upload()
    with gr.Tab("Transcribe from microphone"):
        create_interface_for_microphone()

demo.launch(server_name='0.0.0.0', server_port=1500, ssl_verify=False, share=True)
