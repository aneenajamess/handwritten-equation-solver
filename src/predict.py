import cv2
import numpy as np

from tensorflow.keras.models import load_model

from preprocess import preprocess_image,get_contours,merging_boxes


# Load trained model
digit_model = load_model("models/digit_model.keras")
operator_model=load_model("models/operator_model.keras")

operator_map={0:"+",1:"-",2:"/",3:"=",4:"x"}

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



def predict_digit(character):
    
    image = character.reshape(1,28,28,1)

    prediction = digit_model.predict(
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

def predict_operator(character):

    image = character.reshape(1,28,28,1)

    prediction = operator_model.predict(
        image,
        verbose=0
    )

    operator = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    return operator_map[operator], confidence

def classify_character(character):

    digit, digit_confidence = predict_digit(character)


    operator, operator_confidence = predict_operator(character)
    
    if operator_confidence > digit_confidence:
        return operator, operator_confidence, "operator"

    else:
        return digit, digit_confidence, "digit"



if __name__ == "__main__":

    image = cv2.imread("images/equation.jpeg")


    # preprocessing
    binary = preprocess_image(image)


    # get boxes
    boxes, output = get_contours(binary,image)

    # merge boxes
    boxes = merging_boxes(boxes)

    predictions=[]

    for x,y,w,h in boxes:

        # crop
        character = binary[y:y+h,x:x+w]


        # prepare
        character = prepare_character(character)


        symbol, confidence, symbol_type = classify_character(character)



        predictions.append(symbol)


        print(
            "Prediction:",
            symbol,
            "Confidence:",
            round(float(confidence),3)
        )


        cv2.imshow(
            "character",
            character
        )

        cv2.waitKey(500)



    print("\nFinal equation characters:")

    for i in predictions:
        print(i,end="")


    cv2.destroyAllWindows()

equation = "".join(
    str(symbol)
    for symbol in predictions
)
equation = equation.replace("=","")
print(eval(equation))
'''
res=0
number=0
result=0
if predictions[0] in ['+','-','=','*','x','/']:
    print("Equation may be wrong")
    
equation=[]
for i in range(0,len(predictions),1):
    if predictions[i] in [0,1,2,3,4,5,6,7,8,9]:
        result=(result*10)+predictions[i]
        if(i+1)<len(predictions):
            if predictions[i+1] in ['+','-','=','*','x','/']:
                number=result
                result=0
            if predictions[i] in ['+','-','=','*','x','/']:
                if predictions[i]=='+':
                    res=res+number
                    number=0
                if predictions[i]=='-':
                    res=res-number
                    number=0
                if predictions[i]=='*':
                    res=res*number
                    number=0
                if predictions[i]=='x':
                    res=res*number
                    number=0
                if predictions[i]=='/':
                   res=res/number
                   number=0
        
                if predictions[i]=='=':
                   print(res)
'''
    