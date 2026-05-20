InSight: A Braille Reading Tracking Device

Braille readers and their teachers have a hard time pinpointing the exact struggle areas of the students' reading, so InSight uses a data-driven approach to alleviate this classroom issue

Measures:
1) Speed in characters per second and minute
2) Regressions (reversals): amount of characters re-read
3) Skipped and Unread sections

Outputs:
1) Speed visualization heat map overlayed onto text image
2) CSV File containing raw data to create further visualization if needed

Impact: helping teachers gain insights into the students' reading!


CODE FLOW:

1) Calibrates - using first character of line 1, second character of line 1, last character of line 1, and first character of line 2
it can calibrate the camera coordinates to the coordinates of the paper so that the phone can be placed
at any distance from the paper and still work
2) Snapshot - takes a picture of the text to overlay the heat map later
3) Ready - countsdown to start the reading session once the reader or teacher taps the screen
4) Reading - Student reads and app displays speed in real time in a bottom banner and logs data
5) End - Student taps screen once done, and the app generates heat map and real data CSV
this is the color coded heat map showing speed and regressions
