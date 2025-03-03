import subprocess
import os
import glob


def run_starshadow_py():
    processes = []

    folder = "StarShadow"
    folderAnalysis = folder + "Analysis"
    if not os.path.exists(folderAnalysis):
        os.makedirs(folderAnalysis)

    LCdatas = glob.glob("./" + folder + "/*.txt")

    for index, LCdata in enumerate(LCdatas[2610:2620]):
        print("\n", index, LCdata, len(LCdata))

        try:
            starFolder = os.path.basename(LCdata)[:-4] + "_analysis"

            if not os.path.exists(folderAnalysis + "/" + starFolder + "/" + starFolder + "_summary.csv"):
                process = subprocess.Popen(["python3", "starshadow.py",folderAnalysis, LCdata])
                processes.append(process)
            else:
                print("LC file already exists ", folderAnalysis + "/" + starFolder)

        except Exception as e:
            print(f"{e} Error")

    print(f"Process count: {len(processes)}")

    # Tüm süreçlerin tamamlanmasını bekle
    for i, process in enumerate(processes):
        process.wait()
        print(f"Process {i + 1}/{len(processes)} finished")

if __name__ == "__main__":
    run_starshadow_py()
