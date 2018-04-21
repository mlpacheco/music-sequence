import csv
import wikipedia
import json

summaries = {}
artist_map = {}


with open('data/artist_map.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        artist_map[row[0]] = row[1]

        print row[1]

        try:
            summaries[row[0]] = wikipedia.summary(row[1])
        except wikipedia.exceptions.DisambiguationError:
            try:
                summaries[row[0]] = wikipedia.summary("{0} (band)".format(row[1]))
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue
        except wikipedia.exceptions.PageError:
            continue


with open("artist_map.json", "w") as f:
    json.dump(artist_map, f)

with open("summaries.json", "w") as f:
    json.dump(summaries, f)
