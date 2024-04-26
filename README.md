# Chatbot-Music-Recommendation-System

So first of all, download these files I have pushed into this repository, and save them in some preferred directory. Open VSCode, and open this directory there so you should see the files in the left side panel now. 

Now, it's a healthy practice to run the project in a virtual environment so that it any installations that you do may not affect or interfere with the existing installations on your system. So open a new terminal in VSCode, and type the command <code>python -m venv env</code>. Note that <code>env</code> here is just a name for the environment, if you so wish you can give it any other name. To activate the environment, type the command <code>env\Scripts\activate</code>. You should see (env) in the terminal in the left side.

Now you need to install all the needed libraries. I have included a requirements.txt file which lists all the things needed. All you have to do is run the command <code>pip install -r requirements.txt</code> and all the required files and dependencies will be installed into your virtual environment. To double check if it worked, just navigate to env/Lib from the left side panel on VSCode and you should see all the installed things. Also for the future, whenever you are working on your part of the project and it looks like you will have to install new libraries or dependencies, install those and after the installation, run one additional command <code>pip freeze > requirements.txt</code>. This will update the requirement.txt file and when we pull from the repo the next time, we will again run <code>pip install -r requirements.txt</code> and install any new libraries involved. 

Finally, you can run the application! Run the command <code>uvicorn main:app --reload</code>.
