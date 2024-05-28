import huggingface_hub
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import trange
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cpu")

df = pd.read_csv("sentimentdataset.csv", delimiter=',', header=0,
                 names=['', 'Unnamed: 0', 'Text', 'Sentiment', 'Timestamp', 'User', 'Platform', 'Hashtags', 'Retweets',
                        'Likes', 'Country', 'Year', 'Month', 'Day', 'Hour'])


labelencoder = LabelEncoder()
df['Sentiment'] = df['Sentiment'].str.strip()
df['sentiment_enc'] = labelencoder.fit_transform(df['Sentiment'])
df.rename(columns={'Sentiment': 'sentiment_desc'}, inplace=True)
df.rename(columns={'sentiment_enc': 'sentiment'}, inplace=True)
print(df[['sentiment_desc', 'sentiment']].drop_duplicates(keep='first'))
## create label and sentence list
texts = df.Text.values

# check distribution of data based on labels
print("Distribution of data based on labels: ", df.sentiment.value_counts())

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding='longest')
input_ids = [tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True) for text in
             texts]
labels = df.sentiment.values

## Create attention mask
attention_masks = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=41,
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=41, test_size=0.1)

# convert all our data into torch tensors, required data type for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=df['sentiment_desc'].nunique()).to(device)

# Parameters:
lr = 2e-5
adam_epsilon = 1e-8

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3

num_warmup_steps = 0
num_training_steps = len(train_dataloader) * epochs

### In Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr, eps=adam_epsilon, correct_bias=False,
                  no_deprecation_warning=True)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)  # PyTorch scheduler

## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []

# Gradients gets accumulated by default
model.zero_grad()
unique_label_desc = df['sentiment_desc'].drop_duplicates(keep='first').tolist()
unique_label = df['sentiment'].drop_duplicates(keep='first').tolist()

# Create a dictionary mapping label_desc to label
dictSentiments = dict(zip(unique_label_desc, unique_label))
# tnrange is a tqdm wrapper around the normal python range
for _ in trange(1, epochs + 1, desc='Epoch'):
    print("<" + "=" * 22 + F" Epoch {_} " + "=" * 22 + ">")
    # Calculate total loss for this epoch
    batch_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.type(torch.LongTensor)
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        batch_loss += loss.item()

    avg_train_loss = batch_loss / len(train_dataloader)

    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ", param_group['lr'])
        learning_rate.append(param_group['lr'])

    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage Training loss: {avg_train_loss}')
    model.eval()
    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputsVal = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputsVal[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        tmp_eval_accuracy = accuracy_score(labels_flat, pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1

        df_metrics = pd.DataFrame({'Epoch': epochs, 'Actual_class': labels_flat, 'Predicted_class': pred_flat})

        label2int = {}
        lista = df_metrics['Actual_class'].drop_duplicates(keep='first').tolist()
        lista2 = df_metrics['Predicted_class'].drop_duplicates(keep='first').tolist()
        merged_list = list(set(lista) | set(lista2))
        for valueInt in merged_list:
            if valueInt in dictSentiments.values():
                label2int[next(key for key, value in dictSentiments.items() if value == valueInt)] = valueInt
        print(df_metrics['Actual_class'].values)
        print(df_metrics['Predicted_class'].values)
        print(classification_report(df_metrics['Actual_class'].values, df_metrics['Predicted_class'].values,
                                    target_names=label2int.keys(), digits=len(label2int)))

    print(F'\n\tValidation Accuracy: {eval_accuracy / nb_eval_steps}')
    print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy / nb_eval_steps}')

huggingface_hub.login("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")
model.push_to_hub("finetuned-bert-test")
tokenizer.push_to_hub("finetuned-bert-test")
