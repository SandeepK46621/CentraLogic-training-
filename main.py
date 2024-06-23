
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()


UPLOAD_DIR = "uploaded_files"

os.makedirs(UPLOAD_DIR, exist_ok=True)

def create_text_file(filename, content):
    try:
        with open(filename, 'w') as file:
            file.write(content)
        print(f"File '{filename}' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
            buffer.write(file.file.read())
            whisper_result = pipe("upload/"+file.filename, return_timestamps=True)
            val= str(whisper_result['chunks'][0]['timestamp'])
            pegasus_result=pegasus_summarization.predict(whisper_result['chunks'][0]['text'])
            val=val + pegasus_result

            create_text_file(file.filename,val)

        return JSONResponse(status_code=201, content={"message": "File uploaded successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


    my_var = os.environ.get('HF_TOKEN')
    os.environ['MY_VAR'] = "my_var"

    run= True;

    try:

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        whisper_model_id = "openai/whisper-large-v3"

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        whisper_model.to(device)

        processor = AutoProcessor.from_pretrained(whisper_model_id)

        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )


        pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
        pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")

        pegasus_summarization= pipeline("translation", model=pegasus_model,tokenizer=pegasus_tokenizer)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        

