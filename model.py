import torch
from msclap import CLAP


def init_model(training_type="Projection Finetune", weights_path=None):
    assert training_type in ["Full Finetune", "Projection Finetune", "Eval Mode"]
    clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())

    if weights_path is not None:
        clap_model.clap.load_state_dict(torch.load(weights_path))

    if training_type == "Full Finetune":
        for param in clap_model.clap.audio_encoder.parameters():
            param.requires_grad = True
    elif training_type == "Projection Finetune":
        for param in clap_model.clap.audio_encoder.parameters():
            param.requires_grad = False
        for param in clap_model.clap.caption_encoder.projection.parameters():
            param.requires_grad = True
    else:
        for param in clap_model.clap.audio_encoder.parameters():
            param.requires_grad = False

    return clap_model


def preprocess_text(model, text_queries):
    r"""Load list of class labels and return tokenized text"""
    tokenized_texts = []
    for ttext in text_queries:
        if "gpt" in model.args.text_model:
            ttext = ttext + " <|endoftext|>"
        tok = model.tokenizer.encode_plus(
            text=ttext,
            add_special_tokens=True,
            max_length=model.args.text_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        for key in model.token_keys:
            tok[key] = (
                tok[key].reshape(-1).cuda()
                if model.use_cuda and torch.cuda.is_available()
                else tok[key].reshape(-1)
            )
        tokenized_texts.append(tok)
    return model.default_collate(tokenized_texts)


def get_embeds(model, audio, text):
    audio = model.preprocess_audio(audio, resample=True)
    audio = audio.reshape(audio.shape[0], audio.shape[2])
    audio_embeds = model.clap.audio_encoder(audio)[0]

    text = preprocess_text(model, text)
    text_embeds = model.clap.caption_encoder(text)

    return audio_embeds, text_embeds
