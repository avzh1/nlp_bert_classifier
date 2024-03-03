import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def performance_metrics(trainer, dataloader):
    predicted = []
    expected = []

    predictions = trainer.predict(dataloader)
    predictions = predictions.label_ids
    
    expected = np.array(dataloader.data['labels'])
    expected = expected.astype(int)

    print(classification_report(expected, predictions))

    set(predicted)
    cm = confusion_matrix(expected, predictions, labels=[0, 1]) 

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True));  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.set_title('Confusion Matrix')

    ax.xaxis.set_ticklabels(['Not Patronizing', 'Patronizing'])
    ax.yaxis.set_ticklabels(['Not Patronizing', 'Patronizing'])

    plt.show()


# def model_prediction(model):
#     """Given a model and input for the forward pass, will output pretty-printed output"""
#     # Get one random sample from the test data
#     random_index = X_val.sample(n=1).index.values[0]

#     while random_index not in train_data.index:
#         random_index = X_val.sample(n=1).index.values[0]

#     # Extract random sample
#     tdf = train_data.loc[[random_index]][['text', 'label']]
#     test_paragraph, real_label = tdf['text'].values[0], tdf['label'].values[0]

#     # Get model prediction
#     predicted_label, _ = model.predict([tdf['text'].values[0]])

#     print(f"=== Paragraph example ===\n{test_paragraph.capitalize()}")
#     print(f"\tReal Label:      {real_label}")
#     print(f"\tModel Prediction:{predicted_label[0]}")
