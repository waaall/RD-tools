# RD-tools

this is a modular app that can handle some repetitive tasks in the R&amp;D process

## Functions


### DICOM Processing
Module Function: Converts medical imaging DICOM files into images or videos, saving them in folders mirroring the original DICOMDIR hierarchy.

#### UI Workflow
Select the working directory, which should be the grandparent directory of the DICOMDIR file.
Choose sequence numbers (parent directories of DICOMDIR files). Multiple selections (for multi-experiment scenarios) can be made with space-separated entries.
Click "Extract Selected Sequences".
Navigate to the DICOM Processing option in the left sidebar under File Operations.
Click the "DICOM Processing" button at the bottom-right.
Monitor the log below the button. A "SUCCESS! Log file saved." message indicates completion.

#### CLI Workflow
Run the dicom_to_imgs.py script (located in the modules folder) directly in the terminal. This mirrors the UI logic.

Customization: If your DICOM file structure differs from the assumed hierarchy, rewrite the main function in dicom_to_imgs.py by leveraging the DicomToImage class and base functions in file_basic.py to tailor processing for your specific structure.

### ECG Signal Processing
Module Function: Processes single-lead ECG data (saved as CSV files) from an electrode, generating the following outputs:

Raw data: Time-domain and frequency-domain plots.
Primary filtered data: Time-domain, frequency-domain, and comparison plots.
Advanced processing:
- R-wave peak detection.
- PQRST waveform detection.
- Heart rate calculation.

Steps:
Set the sampling rate (default: 1000 Hz) to match your data.
Select the parent directory of the ECG data folder as the working directory.
Choose the target folder(s) by sequence number.


### Bilibili Video Export
Function: Batch exports cached Bilibili app videos into MP4 format (similar to yt-dlp but optimized for cached files).
Reference: BilibiliCacheVideoMergePython.


### Caption Generation & Summarization
Core: Modified implementation based on VideoCaptioner.
Features:

Supports local configuration of whisper models (e.g., faster-whisper for PotPlayer).
Requires manual setup of dependencies:
whisper-cpp (Linux/Mac recommended).
faster-whisper (Windows recommended).
Limitations:
Translation functionality is removed.
Summarization (using large language models) is pending completion.


### RGB Channel Decomposition/Synthesis
Functions:
- SplitColors: Separates RGB channels of images (useful for fluorescence-labeled images).
- MergeColors: Merges specified channels (e.g., R+G) from image pairs into composite images.


### Com_Driver & SerialPlot
Core: A serial communication base class (Com_Driver) and real-time plotting module (SerialPlot).
Status: Development paused after discovering Serial Studio.
Future Plan: Integrate with the ECG module to build a lightweight ECG visualization tool, if time permits.

### TwistImgs
Function: Applies perspective distortion to images, creating quadrilateral visual effects (useful for custom poster designs or scene mockups).
Note: Specialized for niche use cases.


### more is comming


## How to install

1. Install [Python](https://www.python.org/downloads/) (make sure to add it to the environment variables).
2. (Optional) Create and activate a Python virtual environment.
3. install python third party libs by running the following command in terminal (select1)

```bash
python install.py
```

4. Install [ffmpeg](https://www.ffmpeg.org/download.html) (make sure to add it to the environment variables).
5. install [git](https://git-scm.com/downloads) (make sure to add it to the environment variables).
6. open terminal in a folder you like, and running the following command in terminal

```bash
git clone https://github.com/waaall/RD-tools.git
```

## How to update

open terminal in the `RD-tools` folder, and running the following command in terminal

```bash
git pull
```

NOTICE: if you change the code, maybe have errors when you running `git pull` command. You need to learn how to fix the git conflict.Since you have modified the code, I assume you possess many fundamental skills, such as being proficient in Git and Python; otherwise, you can reinstall.

## How to Use

 start this app by running the following command in terminal. 

```bash
python main.py
```

For more details, please refer to the Help window within the app.
