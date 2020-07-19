import face_recognition
import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', type=str, default='/Users/SirJerrick/Desktop/images')
parser.add_argument('--names', nargs='+', default=None )
args = parser.parse_args()


video_capture = cv2.VideoCapture(0)


def image_encoding(path):
    '''

    :param path: str to path of face image or images
    :return: encoding of image

    '''

    known_face_encodings = []

    for root, dirs, filename in os.walk(path):
        for file in filename:
            if file.endswith(".jpg") or file.endswith(".png"):
                new_path = os.path.join(root, file)
                face_image = face_recognition.load_image_file(new_path)
                face_encoding = face_recognition.face_encodings(face_image)[0]


                known_face_encodings.append(face_encoding)

                return known_face_encodings



def classify():

    known_face_encodings = image_encoding(args.image_dir)

    known_face_names = args.names

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame



        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4


            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():

    classify()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
