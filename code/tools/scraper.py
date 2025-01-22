import csv
import requests

SourceURL = "https://raw.githubusercontent.com/cosmonautinspace/ANN-Autostretcher-AIBAS-Project/main/data/scrapeFromHere/forScraping.csv"

Output = "../../data/scrapedData/scrapedData.csv"

def scrape_and_save_csv():
    try:
        response = requests.get(SourceURL)
        csv_content = response.text.strip().splitlines()
        reader = csv.reader(csv_content)

        with open(Output, mode="w", newline="") as file:
            writer = csv.writer(file)
            for row in reader:
                writer.writerow(row)
        
        print(f"Successfully scraped the data!")
    
    except requests.RequestException as e:
        print(f"Error fetching CSV data: {e}")
    except Exception as e:
        print(f"Error processing or saving CSV data: {e}")

scrape_and_save_csv()
