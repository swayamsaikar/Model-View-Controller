import numpy as np
import PIL.ImageOps
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Now we have to load the npz image files
# This X variable will be used as our input(training)
X = np.load("image.npz")["arr_0"]

# This Y Variable will be our result set
Y = pd.read_csv("data.csv")["labels"]

# This value_counts() will show the number of occurances of each label in the data
print(pd.Series(Y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
           "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
num_classes = len(classes)

# Now we have to split the data into train and test sets
train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, train_size=7500, test_size=2500)

# Now we have to scale the train_X and test_X
# so to scale the images we cannot use StandardScaler()
# so to scale the data we have to divide the train_X and test_X each by 255.0
train_X = train_X/255.0
test_X = test_X/255.0

# (train_X)
# (test_X)

# now its time to train our model
model = LogisticRegression(
    solver='saga', multi_class='multinomial', random_state=9)
model.fit(train_X, train_Y)


def predictAlphabet(image):

    #!!!! I HAVE JUST COPIED THE COMMENTS FROM MY PREVIOUS ALPHABED DETECTION PROJECT !!!!

    # opening the image
    im_pil = Image.open(image)

    # converting to grayscale image - 'L' format means each pixel is
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')

    # resizing the image and also pixalating the image
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

    # applying a pixel filter to filter the pixalated image
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)

    # clipping the image here (cutting or taking a particular part of the image)
    image_bw_resized_inverted_scaled = np.clip(
        image_bw_resized-min_pixel, 0, 255)

    # taking the maximum pixel from the inverted image
    max_pixel = np.max(image_bw_resized)

    image_bw_resized_inverted_scaled = np.asarray(
        image_bw_resized_inverted_scaled)/max_pixel

    # reshape() function returns the same array but with a new shape
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)

    test_pred = model.predict(test_sample)
    return test_pred[0]


'''
!!!!!! I cannot solve this problem :  X has 784 features, but LogisticRegression is expecting 660 features as input !!!!
'''
