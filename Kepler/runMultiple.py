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

    for index, LCdata in enumerate(LCdatas[1570:1575]):
        print("\n", index, LCdata, len(LCdata))

        try:
            fileName = os.path.basename(LCdata)[:-4] + "_analysis"

            if not os.path.exists(folderAnalysis + "/" + fileName):
                process = subprocess.Popen(["python3", "starshadow.py",folderAnalysis, LCdata])
                processes.append(process)
            else:
                print("LC file already exists ", folderAnalysis + "/" + fileName)

        except Exception as e:
            print(f"{e} Error")

    print(f"Process count: {len(processes)}")

    # Tüm süreçlerin tamamlanmasını bekle
    for i, process in enumerate(processes):
        process.wait()
        print(f"Process {i + 1}/{len(processes)} finished")

if __name__ == "__main__":
    run_starshadow_py()
