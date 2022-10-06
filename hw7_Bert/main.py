from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast

from utils import get_device
from data import get_data_loader
from script import train, test

fp16_training = True

accelerator, device = get_device(fp16_training)

model_name = "hfl/chinese-roberta-wwm-ext"
model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)

train_loader, dev_loader, test_loader, dev_questions, test_questions = get_data_loader(tokenizer)

# For large model finetune
# train_loader, dev_loader, test_loader, dev_questions, test_questions = get_data_loader(tokenizer, train_batch_size=2)

if __name__ == '__main__':
    num_epoch = 1
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 

    model.train()

    train(learning_rate, num_epoch, model, optimizer, device, fp16_training, accelerator, train_loader, dev_loader, dev_questions, tokenizer)

    # Save a model and its configuration file to the directory 「saved_model」 
    # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
    # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
    print("Saving Model ...")
    model_save_dir = "saved_model" 
    model.save_pretrained(model_save_dir)

    test(model, test_loader, device, test_questions, tokenizer)