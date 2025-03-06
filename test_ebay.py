from ebaysdk.finding import Connection

api = Connection(
    appid="JordanLi-SportsCa-SBX-fdec4f011-f42bd1af",
    devid="889db8e2-9d05-4405-848f-1bea7f0c4f52",
    certid="SBX-dec4f0113f58-0622-4b6a-9607-1cd3",
    config_file=None,
    siteid="EBAY-US"
)

response = api.execute('findItemsAdvanced', {'keywords': 'Michael Jordan'})
print(response.reply)