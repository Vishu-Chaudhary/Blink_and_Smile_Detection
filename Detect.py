# neccessary imports
import cv2
import imutils
import numpy as np
import dlib
from scipy.spatial import distance as dist
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 60)),
    ("inner_mouth", (61, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 4
MOUTH_AR_THRESH = 0.35
MOUTH_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
COUNTER_MOUTH = 0


# EAR
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    B = dist.euclidean(mouth[1], mouth[11])
    C = dist.euclidean(mouth[2], mouth[10])
    D = dist.euclidean(mouth[3], mouth[9])
    E = dist.euclidean(mouth[4], mouth[8])
    F = dist.euclidean(mouth[5], mouth[7])
    mar = (B+C+D+E+F)/(5*A)
    print mar
    return mar


# Function for creating landmark coordinate list
def land2coords(landmarks, dtype="int"):
    # initialize the list of tuples
    # (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (a, b)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # return the list of (a, b)-coordinates
    return coords


# main Function
if __name__ == "__main__":
    # loading dlib's Hog Based face detector
    face_detector = dlib.get_frontal_face_detector()

    # loading dlib's 68 points-shape-predictor
    # get file:shape_predictor_68_face_landmarks.dat from
    # link: https://github.com/Vishu-Chaudhary/shape-predictor
    landmark_predictor = dlib.shape_predictor('data')

    # 0 means your default web cam
    vid = cv2.VideoCapture(0)
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = FACIAL_LANDMARKS_IDXS["mouth"]
    (ImStart, ImEnd) = FACIAL_LANDMARKS_IDXS["inner_mouth"]

    while True:
        _, frame = vid.read()

        # resizing frame
        # you can use cv2.resize but I recommend imutils because its easy to use
        frame = imutils.resize(frame, width=400)

        # grayscale conversion of image because it is computationally efficient
        # to perform operations on single channeled (grayscale) image
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting faces
        face_boundaries = face_detector(frame_gray, 0)

        for (enum, face) in enumerate(face_boundaries):
            # let's first draw a rectangle on the face portion of image
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 160, 230), 2)

            # Now when we have our ROI(face area) let's
            # predict and draw landmarks
            landmarks = landmark_predictor(frame_gray, face)
            landmarks = land2coords(landmarks)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]
            mouth = landmarks[mStart:mEnd]
            inner_mouth = landmarks[ImStart:ImEnd]
            inner_aspect_ratio(inner_mouth)
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            MAR = mouth_aspect_ratio(mouth)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            innerHull = cv2.convexHull(inner_mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), -1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), -1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), -1)
            cv2.drawContours(frame, [innerHull], -1, (0, 255, 0), -1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if MAR < MOUTH_AR_THRESH:
                COUNTER_MOUTH+=1

                if COUNTER_MOUTH >= MOUTH_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Face :{}".format(enum + 1) + "-Smiling", (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # cv2.putText(frame, "MAR: {:.2f}".format(MAR), (275, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Face :{}".format(enum + 1) + "-notSmiling", (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (125, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        #  Stop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
