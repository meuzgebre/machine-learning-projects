import requests
import urllib3
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import time


def scraper():

	# Initializing driver
	driver = webdriver.Chrome("C:/Users/Mo/chromedriver.exe")	

	url = 'https://db.aa419.org/fakebankslist.php'

	rows = []
	titles = []

	# n = 20 How many pages to go to
	for i in range(7000):

		# Get the appropriet url
		if i == 0: req = driver.get(url)
		else: req = driver.get(url + '?start=' + str(i * 20))

		# Getting Page Content
		src = driver.page_source
		soup = BeautifulSoup(src, 'lxml')

		# Getting the table element
		table = soup.find('table', attrs={'class': 'ewTable'})	
		html_rows = table.find_all('tr')		

		# Getting the titles
		for i in html_rows[0].find_all('th'):
			titles.append(i.text)

		# Getting the cell values
		for row in html_rows:
			cells = []
			for cell in row.find_all('td'):			
				cells.append(cell.text)
			rows.append(cells)		

		# Navigaing to the next page
		# driver.find_element_by_xpath("//div[@id='main']/a[@*][13]").click		


	# Creating the dataframe
	df = pd.DataFrame(rows, columns=list('abcdefg'))
	print(df.head())
	df.to_csv('fake url db.csv', index=False)




if __name__ == '__main__':
	scraper()