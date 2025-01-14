# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC asr.Transcribe server."""

from concurrent import futures
import logging

import grpc
import protos.asr_pb2 as asr_pb2
import protos.asr_pb2_grpc as asr_pb2_grpc

import torch
import torchaudio
from faster_whisper import WhisperModel

import librosa
import numpy as np

class fast_pipeline():
    def __init__(self, model):
        self.model = model
        self.epd_margin = 0.1

    def __call__(self, filepath, generation_config, duration):
        if "condition_on_prev_tokens" in generation_config:
            generation_config["condition_on_previous_text"] = generation_config.pop("condition_on_prev_tokens")
        if "logprob_threshold" in generation_config:
            generation_config["log_prob_threshold"] = generation_config.pop("logprob_threshold")

        #generation_config["initial_prompt"] = '추론해야 할 음성은 약 10미터 거리에서 약간의 화이트 노이즈가 있는 국립 극장 환경에서 녹음된 음성입니다. 음성에는 ‘테러’, ‘폭발’, ‘북한’, ‘타격’, ‘전쟁’, ‘도발’, ‘공습’, ‘폭음’, ‘총성’, ‘지뢰’, ‘폭탄’, ‘위협’과 같은 키워드를 포함한 위협적인 문장이 발화됩니다. 화자의 어조는 긴장감이 감돌며, 이러한 용어들이 암시하는 잠재적 위험을 전달하고 있습니다. 이러한 조건을 바탕으로 음성인식을 잘 수행해 주세요.'
        #generation_config["initial_prompt"] = '10미터 거리에서 국립극장의 약간의 화이트 노이즈가 있는 환경에서 녹음된 음성입니다. 음성에는 ‘테러’, ‘폭발’, ‘북한’, ‘전쟁’, ‘도발’, ‘공습’, ‘총성’, ‘지뢰’, ‘폭탄’등과 같은 키워드를 포함한 위협적인 문장이 발화됩니다. 화자의 어조는 긴장감이 감돌며, 이러한 용어들이 암시하는 잠재적 위험을 전달하고 있습니다.'
        #generation_config["initial_prompt"] = '10미터 거리의 약간의 화이트 노이즈가 있는 국립 극장 환경에서 ‘테러’, ‘폭발’, ‘북한’, ‘타격’, ‘전쟁’, ‘도발’, ‘공습’, ‘폭음’, ‘총성’, ‘지뢰’, ‘폭탄’ 등의 위협적인 키워드를 포함한 문장이 발화된 음성입니다.'
        #generation_config["initial_prompt"] = '10미터 거리의 약간의 화이트 노이즈가 있는 국립 극장 환경에서 위협적인 키워드를 포함한 문장이 발화된 음성입니다.'
        
        # Not to use power augmentation, uncomment below.
        # segments, _ = self.model.transcribe(filepath, beam_size=1, without_timestamps=False, **generation_config)
        ##### This code is for power augmentation & multi-channel #####
        waveform = None
        if 1:
            waveform, audio_sf = librosa.load(filepath)     # waveform type : <numpy.ndarray>
            #waveform, audio_sf = torchaudio.load(filepath)
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            elif not isinstance(waveform, torch.Tensor):
                waveform = decode_audio(waveform, sampling_rate=16000)

            if audio_sf != 16000:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=audio_sf, new_freq=16000
                )

            #print('*' * 20)
            #print(waveform.shape)

            if len(waveform.shape) == 1:  # single channel
                waveform = waveform.unsqueeze(0)  # [samples] to [1, samples]
            elif len(waveform.shape) == 2:  # 2D tensor / multi channel
                #waveform = waveform.mean(0).unsqueeze(0)  # multi to single and to [1, samples]
                if waveform.shape[0] == 16:  # Check if it's a 16-channel audio
                    selected_channels = waveform[[0, 2, 4, 6], :]  # Select channels 0, 2, 4, 6
                    waveform = selected_channels.mean(0).unsqueeze(0)  # Average the selected channels and keep shape [1, samples]
                else:
                    waveform = waveform.mean(0).unsqueeze(0)  # For other multi-channel audio, reduce to mono

            #print('*' * 20)
            #print(waveform.shape)
            
            gain = 10
            waveform = waveform * gain

        torchaudio.save("test_audio.wav", waveform, 16000)
        filepath = "test_audio.wav"
        
        segments, _ = self.model.transcribe(filepath, beam_size=1, without_timestamps=False, **generation_config)
        ##### This code is for power augmentation & multi-channel #####

        '''
        segments, _ = self.model.transcribe(filepath if waveform is None else waveform, 
                                            beam_size=1, 
                                            without_timestamps=False, 
                                            **generation_config
                                    )
        '''
        notimestamped_text = ""
        timestamped_text = ""
        for segment in segments:
            if float(segment.start) >= float(duration) - self.epd_margin:
                break
            notimestamped_text += f"{segment.text}\n"
            timestamped_text += f"[{segment.start:.2f}:{segment.end:.2f}] {segment.text}\n"

        return notimestamped_text, timestamped_text

class ASR(asr_pb2_grpc.ASRServiceServicer):
    def __init__(self):
        # Initialize and load the model and pipeline here, so it's done only once
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32
        self.use_model("prompt_nat_v3")
        #self.use_model("sitec_ft_v3")
        #self.use_model("vanilla_v3")


        # Decoding configurations
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        no_speech_threshold = 0.6
        logprob_threshold = -1.0

        self.generate_config = {"task": "transcribe"}
        self.generate_config["max_new_tokens"] = 256
        self.generate_config["condition_on_prev_tokens"] = False
        self.generate_config["no_speech_threshold"] = no_speech_threshold
        self.generate_config["temperature"] = temperature
        self.generate_config["logprob_threshold"] = logprob_threshold

    def use_model(self, model_choice):
        model_path = f"/home/jieun/workspace/whisper_gradio_demo/models/{model_choice}"
        #model_path = f"/home/mozi/Workspace/whisper_gradio_demo/models/{model_choice}"
        self.model = WhisperModel(model_path, device="cuda", compute_type="float16")
        self.pipe = fast_pipeline(model=self.model)

    def transcribe_speech(self, filepath, language, with_timestamps, duration):
        if language == "ko":
            self.generate_config["no_speech_threshold"] = 0.55
            self.generate_config["logprob_threshold"] = -0.3
        else:
            self.generate_config["no_speech_threshold"] = 0.6
            self.generate_config["logprob_threshold"] = -1.0
        self.generate_config["language"] = language

        notimestamped_text, timestamped_text = self.pipe(
            filepath,
            self.generate_config,
            duration
        )
        
        return timestamped_text if with_timestamps else notimestamped_text

    def Transcribe(self, request, context):
        transcription = self.transcribe_speech(request.filepath, request.language, request.with_timestamps, request.duration)
        reply = asr_pb2.ASRReply()
        reply.transcription = transcription
        return reply
    
    def ReloadModel(self, request, context):
        try:
            self.use_model(request.model_name)
            return asr_pb2.ReloadModelReply(success=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Failed to reload model: {e}')
            return asr_pb2.ReloadModelReply(success=False)

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asr_service = ASR()  # Create an instance of the ASR service
    asr_pb2_grpc.add_ASRServiceServicer_to_server(asr_service, server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
