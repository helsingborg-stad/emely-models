import requests_async as requests
import asyncio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




"""
This script can test response times for a model (deployed or local) 
in a simulation of X chatters writing simulataneously. 

This is done using asyncio and each round is a battery of X messages that are gathered before the next battery is sent. This

It plots a scatter of elapsed time(y) and message_nbr(x) and colors messages from the same simulated chatter.
It also prints the mean elapsed time for all requests and grouped by chatter.

"""

# ------------ Parameters ----------------------------------

url = "http://localhost:8080/inference"
nbr_runs = 10
nbr_parallell_messages = 4

data = {
    'text': "Hi emely how are you doing today????? I'm fine thanks hihi"
}


# ------------------------------------------------------------------------------

async def send_message():
    r = await requests.post(url, json=data)
    return r


loop = asyncio.get_event_loop()


elapsed = []
for i in range(nbr_runs):

    tasks  = asyncio.gather(*[send_message() for _ in range(nbr_parallell_messages)])
    responses = loop.run_until_complete(tasks)

    for j, r in enumerate(responses):
        elapsed.append(
            {
                'message': i,
                'elapsed': r.elapsed.total_seconds(),
                'user': j,
            }
        )


loop.close()
df = pd.DataFrame(elapsed)
print('Total mean:', df['elapsed'].mean())
print("Means by 'user' ")
print(df.groupby("user").mean())
plt.figure()
sns.scatterplot(x="message", y="elapsed", data=df, hue="user")
plt.show()