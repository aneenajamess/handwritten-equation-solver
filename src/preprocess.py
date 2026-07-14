import cv2

def preprocess_image(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert to grayscale

    blur = cv2.GaussianBlur(gray, (5,5), 0) #remove noise

    #adaptive threshold, chars become white, bg becomes black
    #removes noise 
    thresh = cv2.adaptiveThreshold(
       blur,
       255,
       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
       cv2.THRESH_BINARY_INV,
       17,
       4
    )

    # morphological operations -> used to improve
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # closing =>dilation followed by erosion
    # useful for fixing broken strokes, tiny gaps
    closing = cv2.morphologyEx(
       thresh,
       cv2.MORPH_CLOSE,
       kernel
    )
    return closing


def get_contours(binary,image):
    # find contours
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output=image.copy()

    bounding_boxes=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if w<10 or h<10:
            continue

        bounding_boxes.append((x,y,w,h))
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    
    return bounding_boxes,output

def merge_boxes(box1,box2):
    x1=box1[0]
    y1=box1[1]
    w1=box1[2]
    h1=box1[3]

    x2=box2[0]
    y2=box2[1]
    w2=box2[2]
    h2=box2[3]

    new_x=min(x1,x2)
    new_y=min(y1,y2)
    right=max(x1+w1,x2+w2)
    bottom=max(y1+h1,y2+h2)

    new_w=right-new_x
    new_h=bottom-new_y

    return (new_x, new_y, new_w, new_h)

def should_merge(box1,box2):
    x1=box1[0]
    y1=box1[1]
    w1=box1[2]
    h1=box1[3]

    x2=box2[0]
    y2=box2[1]
    w2=box2[2]
    h2=box2[3]

    x1_centre=x1+w1/2
    y1_centre=y1+h1/2

    x2_centre=x2+w2/2
    y2_centre=y2+h2/2

    x=abs(x2-x1)
    y=abs(y2-y1)

    avg_w=(w1+w2)/2
    avg_h=(h1+h2)/2

    if (x<avg_w and y<avg_h*(1.5)):
        return True
    


def merging_boxes(boxes):
    change=True

    while change:
        change=False
        new_boxes=[]
        used=[False]*len(boxes)

        for i in range(len(boxes)):
            current=boxes[i]
            if used[i] is True:
                continue

            for j in range(i+1,len(boxes),1):
                if used[j]==True:
                    continue

                if should_merge(current,boxes[j]):
                    current=merge_boxes(current,boxes[j])
                    used[j]=True
                    change=True

            new_boxes.append(current)

        boxes = new_boxes

    return boxes
            













if __name__ == "__main__":

    image = cv2.imread("images/equation.jpeg")

    binary = preprocess_image(image)

    boxes, output = get_contours(binary, image)
    print(len(boxes))
    print(boxes)

    cv2.imshow("Binary", binary)
    cv2.imshow("Bounding Boxes", output)
    result=merging_boxes(boxes)
    

    output1 = image.copy()

    for x, y, w, h in result:
        cv2.rectangle(output1,(x, y),(x+w, y+h),(0,255,0),2)

    cv2.imshow("Res", output1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()