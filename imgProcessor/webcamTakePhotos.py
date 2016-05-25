'''
a simple script to see and save photos in a given folder
... in case you can't find your webcam app
'''
if __name__ == '__main__':
    
    import cv2
    import os
    
    SAVE_FOLDER = '.'
    
    
    cap = cv2.VideoCapture(0)
    c = 0
    
    print('press [q] for quit, [s] to save a frame')
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',gray)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print('quit')
            break
        if k == ord('s'):
            c+=1
            fname = os.path.join(SAVE_FOLDER, '%s.png' %c)
            print('save image under ... %s' %fname)
            cv2.imwrite(fname, gray)
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()