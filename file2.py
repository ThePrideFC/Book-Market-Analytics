import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from urllib.parse import urljoin
url = "https://books.toscrape.com"
scraped_books = []
page_counter = 1
while url:
    print(f"Scraping Page {page_counter}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Terminating: Failed to reach page path{url}")
        break
    soup = BeautifulSoup(response.text, "html.parser")
    book_pods = soup.find_all("article", class_="product_pod")
    for pod in book_pods:
        title = pod.h3.a.get("title", pod.h3.a.text)
        price_text = pod.find("p", class_="price_color").text
        rating_classes = pod.find("p", class_="star-rating")["class"]
        rating_word = [c for c in rating_classes if c != "star-rating"][0]
        stock_text = pod.find("p", class_="instock availability").text.strip()
        scraped_books.append({"Title":title,"Raw_Price":price_text,"Raw_Rating":rating_word,"Availability":stock_text})
    next_button = soup.find("li", class_="next")
    if next_button:
        next_page_file = next_button.a["href"]
        url = urljoin(url, next_page_file)
        page_counter += 1
        time.sleep(0.5)
    else:
        url = None
df_raw = pd.DataFrame(scraped_books)
print(f"\n Scraping phase complete. Successfully retrieved {len(df_raw)} records.")
print(df_raw.head())
df_clean = df_raw.copy()
df_clean["Price_GBP"] = df_clean["Raw_Price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
rating_map = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}
df_clean["Rating_Numeric"] = df_clean["Raw_Rating"].map(rating_map)
df_clean["In_Stock"] = df_clean["Availability"].apply(lambda x: 1 if "In stock" in x else 0)
df_clean["Title_Length"] = df_clean["Title"].apply(len)
df_clean_final = df_clean[["Title", "Price_GBP", "Rating_Numeric", "In_Stock", "Title_Length"]]
df_clean_final.to_csv("cleaned_book_market_dataset.csv", index = False)
print("\nMissing values:")
print(df_clean_final.isnull().sum())
print("\n Rows of the data:")
print(df_clean_final.head())
plt.figure(figsize=(8, 5))
sns.barplot(data=df_clean_final, x="Rating_Numeric", y="Price_GBP", hue="Rating_Numeric", palette="Blues", errorbar=None, legend=False)
plt.title("Average Book Price Across Star Ratings", fontsize=14, fontweight="bold")
plt.xlabel("Star Rating Value", fontsize=12)
plt.ylabel("Average Market Price (£)", fontsize=12)
plt.figure(figsize=(6, 5))
numeric_cols = ["Price_GBP", "Rating_Numeric", "In_Stock", "Title_Length"]
correlation_matrix = df_clean_final[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Extracted Matrix Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
x = df_clean_final[["Rating_Numeric", "In_Stock", "Title_Length"]]
y = df_clean_final["Price_GBP"]
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE):     £{mae:.4f}")
print(f"Root Mean Squared Error (RMSE): £{rmse:.4f}")
print(f"Coefficient of Determination (R²): {r2:.4f}")
print("\n---Regression Coefficient Model Breakdown---")
for feature, coef in zip(x.columns, model.coef_):
    print(f"Feature:{feature:<15} | Impact Weight Coefficient: {coef:.4f}")
print(f"Intercept Constant Term Value: {model.intercept_:.4f}")
