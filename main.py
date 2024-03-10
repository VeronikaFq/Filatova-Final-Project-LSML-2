#Import modules, downloading the data 
import os
import neptune
import pandas as pd
import random
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, BertConfig
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request

# Use the Flask web framework
app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/test_params': {"origins": "http://localhost:8080"}})

# Log to a neptune project in order to fix run with new params 
WORKSPACE_NAME = "veronikafilatova95"
PROJECT_NAME = "project-text-classification"
os.environ["NEPTUNE_PROJECT"] = f"{WORKSPACE_NAME}/{PROJECT_NAME}"
project = neptune.init_project(project="veronikafilatova95/project-text-classification", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlM2Q5OWE5Yi00NWZkLTRhODUtYWUzNC1lZmRjN2I5OTU5NzgifQ==")

#Load JSON file into pandas DataFrame for both datasets
train = pd.read_json('train.jsonl', lines=True)
val = pd.read_json('val.jsonl', lines=True)

# Get BERT embeddings from the questions and passages in train dataset and vectorize data.
# We will use bert-base-multilingual-cased model from hugging face library (https://huggingface.co/bert-base-multilingual-cased)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Define an auxiliary function to handle the tokenization process:
# - divide the questions and excerpts into tokens
# - add a sentence start marker and a marker indicating the separation between the question and the passage, as well as the end of the input.
# - match tokens with their IDs.
# - fill in (with a marker) or truncate each pair of questions/excerpts to max_seq_length.
# - create attention masks to distinguish the corresponding markers from the fill markers.
def preprocess (tokenizer, questions, passages):
  input_ids = []
  attention_masks = []
  for question, passage in zip(questions, passages):
    encoded_dict = tokenizer.encode_plus(
          question, passage,                      # Sentences to encode.
          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
          max_length = 256,           # Pad & truncate all sentences.
          pad_to_max_length = True,
          return_attention_mask = True,   # Construct attn. masks.
          return_tensors = 'pt')    # Return pytorch tensors.
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
  return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

#Prepare and tokenize train data
passages = train.question.values
questions = train.tokenized_passage.values
answers = train.label.values.astype(int)
input_ids, attention_masks = preprocess(tokenizer, questions, passages)
labels = torch.tensor(answers)

#Prepare and tokenize validation data
passages_val = val.passage.values
questions_val = val.question.values
answers_val = val.label.values.astype(int)
input_ids_val, attention_masks_val = preprocess(tokenizer, questions_val, passages_val)
labels_val = torch.tensor(answers_val)

# Create test DataLoader for predictions
prediction_data = TensorDataset(input_ids_val, attention_masks_val, labels_val)
prediction_sampler = SequentialSampler(prediction_data)

# Create train and validaton DataLoader for training 
def create_data_loader(batch_size=32):
  # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
  # Create the DataLoaders for our training and validation sets.
  # We'll take training samples in random order.
  train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
  # For validation the order doesn't matter, so we'll just read them sequentially.
  validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
  return train_dataloader, validation_dataloader

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels = 2, # The number of output labels--2 for binary classification.
    output_attentions = False, 
    output_hidden_states = False)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


####
# Function to train the model
def train(model, train_dataloader, validation_dataloader, params, device):
  # (Neptune) Log params
  run["parameters"] = params

  optimizer = AdamW(model.parameters(), lr = params['lr'], eps = params['eps'], 
                  no_deprecation_warning = True)
  epochs = params['epochs']
  
  # Total number of training steps is [number of batches] x [number of epochs].
  total_steps = len(train_dataloader) * epochs
  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)
  model.cuda()
  # Set the seed value all over the place to make this reproducible.
  seed_val = 42
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  
  # We'll store a number of quantities such as training and validation loss, validation accuracy
  training_stats = []

  # For each epoch...
  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Reset the total loss for this epoch.
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Report progress.
            print("Batch {}".format(step), "of {}".format(len(train_dataloader)))
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        res = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = res[0]
        logits = res[1]
        # Accumulate the training loss over all of the batches.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    run["metrics/train/avg_train_loss"].append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # After the completion of each training epoch, measure our performance on our validation set.
    print("")
    print("Running Validation...")
    
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()
    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # The "logits" are the output values prior to applying an activation function like the softmax.
            res = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = res[0]
            logits = res[1]

        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    run["metrics/train/avg_valid_loss"].append(avg_val_loss)
    run["metrics/train/avg_valid_accuracy"].append(avg_val_accuracy)
    # Record all statistics from this epoch.
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
        })
  print("")
  print("Training complete!")
  
###

###
# Function for testing the model and making predictions
def test(model, prediction_dataloader, device):
  # Prediction on test set
  # Put model in evaluation mode
  model.cuda()
  model.eval()
  
  # Tracking variables
  predictions , true_labels = [], []
  # Predict
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

  accuracy_set = []
  # Evaluate each test batch
  print('Calculating accuracy score for each batch...')
  
  # For each input batch...
  for i in range(len(true_labels)):
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  # Calculate and store the coef for this batch.
  accuracy_set.append(accuracy_score(true_labels[i], pred_labels_i))

  # Combine the results across all batches.
  flat_predictions = np.concatenate(predictions, axis=0)
  # For each sample, pick the label (0 or 1) with the higher score.
  flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
  # Combine the correct labels for each batch into a single list.
  flat_true_labels = np.concatenate(true_labels, axis=0)
  # Calculate the accuracy score
  accuracy = accuracy_score(flat_true_labels, flat_predictions)
  run["metrics/test/accuracy "] = accuracy

  return (predictions, accuracy)
  
###

###
# The microservice will accept the model parameters, train the model and return the predicted values and accuracy.
@app.route('/test_params', methods=['POST'])
def test_params_post_request(): 
  params = request.json
  train_dataloader, validation_dataloader = create_data_loader(params['batch_size'])
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=params['batch_size'])
  
  # We will run this model on the GPU
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
    
  # Initialize a neptune prun and model_version
  run = neptune.init_run(project="veronikafilatova95/project-text-classification",
                         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlM2Q5OWE5Yi00NWZkLTRhODUtYWUzNC1lZmRjN2I5OTU5NzgifQ==")
  model_version = neptune.init_model_version(model="PROJ-MOD", project="veronikafilatova95/project-text-classification",
                                             api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlM2Q5OWE5Yi00NWZkLTRhODUtYWUzNC1lZmRjN2I5OTU5NzgifQ==")
  # Train model
  train(model, train_dataloader, validation_dataloader, params, device)

  # Test model
  result = test(model, prediction_dataloader, device)
  predictions = result[0]
  accuracy = result[1]

  # Fetch parameters and metrics to the current run and model
  run_dict = {
    "id": run["sys/id"].fetch(),
    "name": run["sys/name"].fetch(),
    "url": run.get_url(),
    "betch_size": run["parameters/batch_size"].fetch(), 
    "epochs": run["parameters/epochs"].fetch(),
    "eps": run["parameters/eps"].fetch(),
    "lr": run["parameters/lr"].fetch(),
    "test_accuracy": run["metrics/test/accuracy "].fetch()}
  model_version["run"] = run_dict
  model_version_dict = {
    "id": model_version["sys/id"].fetch(),
    "url": model_version.get_url()}
  run["model"] = model_version_dict
  # Put model version on the staging
  model_version.change_stage("staging")

  return jsonify({'Predcirions':predictions, 'Accuracy score':accuracy})

###
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
