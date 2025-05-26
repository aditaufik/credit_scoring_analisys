{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e3df12fc-3c4d-4bc6-ae7b-6382dd148bdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "model = joblib.load(\"model/gboost_model.joblib\")\n",
    "result_target = joblib.load(\"model/encoder_target.joblib\")\n",
    "\n",
    "def prediction(data):\n",
    "    \"\"\"Making prediction\n",
    "\n",
    "    Args:\n",
    "        data (Pandas DataFrame): Dataframe that contain all the preprocessed data\n",
    "\n",
    "    Returns:\n",
    "        str: Prediction result (Good, Standard, or Poor)\n",
    "    \"\"\"\n",
    "    result = model.predict(data)\n",
    "    final_result = result_target.inverse_transform(result)[0]\n",
    "    return final_result"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
