import requests
import urllib3
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd


def scraper():

	# Initializing driver
	driver = webdriver.Chrome("C:/Users/Mo/chromedriver.exe")

	# ML# https://www.indeed.com/jobs?q=machine%20learning&l&vjk=2b46ec3a7eeac740
	# DA # https://www.indeed.com/jobs?q=data%20analyst&l&vjk=e9d986a6bc58ed3f
	# AI # https://www.indeed.com/jobs?q=artificial%20intelligence&l&vjk=21d186efb9d01522
	# DE # https://www.indeed.com/jobs?q=data%20engineer&l&vjk=1cdd3ce1b35d9909

	req = driver.get('https://www.indeed.com/jobs?q=data%20analyst&l&vjk=e9d986a6bc58ed3f')

	blocks = []
	role = []
	company = []
	location = []

	# n = 500 How many pages to go to
	for _ in range(500):

		# Getting Page Content
		src = driver.page_source
		soup = BeautifulSoup(src, 'lxml')

		# Iterating each job description conatiner
		for div in soup.find_all('div', attrs = {"class": "job_seen_beacon"}):
			# blocks.append(div)
			# Forming New Labels
			role.append(div.h2.text)
			company.append(div.find('span', attrs = {"class": "companyName"}).text)
			location.append(div.find('div', attrs = {"class": "companyLocation"}).text)

		# Navigaing to the next page
		driver.find_element_by_xpath('//a[@aria-label="Next"]').click


	# Creating Dataframe
	df = pd.concat([pd.DataFrame(role, columns=['Role']), pd.DataFrame(company, columns=['Company']), pd.DataFrame(location, columns=['Location'])], axis=1)
	
	# Export to CSV
	df.to_csv('ma_job_landing.csv', index=False)
	# print(f'Done! Total{df.shape}')


if __name__ == '__main__':
	scraper()