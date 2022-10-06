import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

def evaluate(tokenizer, data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        # if prob > max_prob:
        # Add condition for wrong (start & end) index and prevent too long answer
        if prob > max_prob and start_index <= end_index and end_index - start_index <= 20: 
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

def train(learning_rate, num_epoch, model, optimizer, device, fp16_training, accelerator, train_loader, dev_loader, dev_questions, tokenizer):  
    validation = True
    logging_step = 900
    print("Start Training ...")
    total_steps = num_epoch * len(train_loader)
    gradient_acc = 8 # batch accumulation parameter
    warm_up_ratio = 0.1    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warm_up_ratio * total_steps, 
                                                num_training_steps = total_steps)

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0
        
        for data in tqdm(train_loader):	
            # Load all data into GPU
            data = [i.to(device) for i in data]
            
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss
            
            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()
            
            if step % gradient_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step() # decay 2
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            # optimizer.param_groups[0]["lr"] = max(0, optimizer.param_groups[0]["lr"] - learning_rate/total_steps) # decay 1
            lr = optimizer.param_groups[0]["lr"]
            
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | lr {lr} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
                    # prediction is correct only if answer text exactly matches
                    dev_acc += evaluate(tokenizer, data, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
            model.train()

def test(model, test_loader, device, test_questions, tokenizer):      
    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                        attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(tokenizer, data, output))

    result_file = "result.csv"
    with open(result_file, 'w') as f:    
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")      