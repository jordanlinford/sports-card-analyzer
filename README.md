 HEAD
# sports-card-analyzer
# eBay Sales Analysis Dashboard

This Streamlit application fetches and analyzes your eBay sold items data from the last 90 days, providing insights and visualizations about your sales performance.

## Features

- Fetches sold items data from the last 90 days
- Displays total sales and average price
- Shows a line chart of sales over time
- Provides raw data in a table format
- Allows downloading the data as CSV

## Setup

1. Install the required dependencies:
```bash
pip install streamlit pandas requests python-dotenv
```

2. Set up your eBay API credentials:
   - Copy the `.env.example` file to `.env`
   - Add your eBay API credentials to the `.env` file:
     ```
     EBAY_APP_ID=your_app_id
     EBAY_DEV_ID=your_dev_id
     EBAY_CERT_ID=your_cert_id
     EBAY_AUTH_TOKEN=your_auth_token
     ```

3. Get your eBay OAuth token:
   - Log in to the [eBay Developer Program](https://developer.ebay.com/)
   - Generate a user token for your application
   - Add the token to your `.env` file

## Running the Application

To run the application, use the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser. Click the "Fetch eBay Data" button to start retrieving and analyzing your sales data.

## Notes

- The application uses the eBay Trading API with the GetSellerList call
- Data is fetched in pages of 200 items each
- The analysis includes only completed (sold) items
- Make sure your eBay API credentials are valid and have the necessary permissions 
 c941d0b (First commit - adding my sports card tracker)
