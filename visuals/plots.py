import numpy as np
import matplotlib.pyplot as plt

def readdata(filename):

    with open(filename, "r") as f:
        return np.genfromtxt(f, delimiter=',', skip_header=1, unpack=False)
    
def summarizeData(plays_csv="out/plays.csv"):

    innings = []
    years = []
    homeruns = []

    for each_row_of_data in dataStats:
        innings.append(each_row_of_data[1])
        years.append()
        

    plt.plot(innings, runs, "o")
    plt.title("Scores by Inning")
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Scoring", fontsize=14)

    plt.show()
        


def HFregression(hf):
    plt.title("Regression Across Innings", fontsize=14)
    plt.xlabel("Innings", fontsize=14)
    plt.ylabel("Average Scorings", fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.plot()