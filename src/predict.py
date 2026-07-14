import cv2
import numpy as np

from tensorflow.keras.models import load_model

from preprocess import preprocess_image,get_contours,merging_boxes


# Load trained model
model = load_model("models/digit_model.keras")


def prepare_character(character):

    # Add padding

    padding = 20

    character = cv2.copyMakeBorder(
        character,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )


    # Resize to MNIST size

    character = cv2.resize(
        character,
        (28,28)
    )


    # Normalize

    character = character.astype(
        "float32"
    ) / 255.0


    return character



def predict_character(character):


    # reshape:
    # (28,28)
    # ->
    # (1,28,28,1)

    image = character.reshape(
        1,
        28,
        28,
        1
    )


    prediction = model.predict(
        image,
        verbose=0
    )


    digit = np.argmax(
        prediction
    )


    confidence = np.max(
        prediction
    )


    return digit, confidence





if __name__ == "__main__":


    image = cv2.imread(
        "images/equation.jpeg"
    )


    # preprocessing

    binary = preprocess_image(
        image
    )


    # get boxes

    boxes, output = get_contours(
        binary,
        image
    )


    # merge boxes

    boxes = merging_boxes(
        boxes
    )


    predictions=[]


    for x,y,w,h in boxes:


        # crop

        character = binary[
            y:y+h,
            x:x+w
        ]


        # prepare

        character = prepare_character(
            character
        )


        # predict

        digit,confidence = predict_character(
            character
        )


        predictions.append(
            digit
        )


        print(
            "Prediction:",
            digit,
            "Confidence:",
            round(float(confidence),3)
        )


        cv2.imshow(
            "character",
            character
        )

        cv2.waitKey(500)



    print("\nFinal equation characters:")

    print(
        predictions
    )


    cv2.destroyAllWindows()