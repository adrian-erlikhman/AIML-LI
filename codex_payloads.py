# Auto-generated payloads for notebooks to copy into Codex environments
"""Utility for emitting notebook JSON strings that can be copy-pasted.
"""

import argparse
from pathlib import Path

NOTEBOOK_JSON = {
    'u01_l01_core_intro_ai_v1.ipynb': '''
{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "194b395d",
      "metadata": {},
      "source": [
        "# Train Your First Classifier (Toy Data, Accuracy & Limits)\n",
        "_Lesson 01 – What is AI? Automation vs. Learning_\\n\\n**Authors:** LACES AIML\\n\\n**Standards Alignment**\\nCA CS Standards: 3A-IC-26; 3A-DA-09; 3B-AP-14\n",
        "LAUSD Digital Citizenship: Privacy & Security; Equity & Access (STEAM/CTE)\\n\\n**Time Estimate:** 25 minutes\\n\\n**Learning Objectives**\\n- Differentiate rule-based automation from machine learning with examples\n",
        "- Run a tiny ML pipeline and interpret a basic metric\n",
        "- Explain one ethical consideration (bias, privacy, or accountability) in context"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "6462e4ff",
      "metadata": {},
      "outputs": [],
      "source": [
        "# Toggle for teacher key\n",
        "SHOW_TEACHER_KEY = False\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.datasets import make_classification\n",
        "import random\n",
        "\n",
        "np.random.seed(42)\n",
        "random.seed(42)\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "\n",
        "%matplotlib inline\n",
        "plt.style.use('seaborn-v0_8-colorblind')\n",
        "plt.rcParams['figure.figsize'] = (6, 4)\n",
        "plt.rcParams['axes.titlesize'] = 12\n",
        "plt.rcParams['axes.labelsize'] = 11"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "701d6e0b",
      "metadata": {},
      "source": [
        "## Concept Primer\n",
        "\n",
        "- **Pipeline:** collect data → split into train/test → train a model → evaluate.\n",
        "- **Accuracy** measures how often predictions match reality.\n",
        "- Small datasets help us see each step clearly before scaling up.\n",
        "\n",
        "```\n",
        "[Data] -> [Split] -> [Model learns] -> [Metrics] -> [Decisions]\n",
        "```\n",
        "\n",
        "Keep track of each stage to debug and explain choices.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "9a42b1bf",
      "metadata": {},
      "source": [
        "## Guided Activity A\n",
        "### Generate and Explore a Tiny Dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "2d48fabc",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "features, labels = make_classification(\n",
        "    n_samples=300,\n",
        "    n_features=2,\n",
        "    n_redundant=0,\n",
        "    n_clusters_per_class=1,\n",
        "    class_sep=1.5,\n",
        "    random_state=42,\n",
        ")\n",
        "df = pd.DataFrame(features, columns=[\"feature_1\", \"feature_2\"])\n",
        "df[\"label\"] = labels\n",
        "print(df.head())\n",
        "\n",
        "plt.figure()\n",
        "for label_value, marker in [(0, 'o'), (1, '^')]:\n",
        "    subset = df[df['label'] == label_value]\n",
        "    plt.scatter(subset['feature_1'], subset['feature_2'], label=f\"Class {label_value}\", alpha=0.7)\n",
        "plt.title(\"Tiny Classification Dataset\")\n",
        "plt.xlabel(\"Feature 1\")\n",
        "plt.ylabel(\"Feature 2\")\n",
        "plt.legend()\n",
        "plt.show()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b8314d81",
      "metadata": {},
      "source": [
        "\n",
        "### Predict Before You Run\n",
        "1. Are the two classes clearly separated?\n",
        "2. Which class seems larger?\n",
        "3. How might overlapping points affect accuracy?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "a06ced24",
      "metadata": {},
      "source": [
        "## Guided Activity B\n",
        "### 🧪 Try It: Train, Predict, Interpret"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "02044d90",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "        X = df[[\"feature_1\", \"feature_2\"]].values\n",
        "        y = df[\"label\"].values\n",
        "\n",
        "        X_train, X_test, y_train, y_test = train_test_split(\n",
        "            X, y, test_size=0.25, random_state=42, stratify=y\n",
        "        )\n",
        "\n",
        "        model = LogisticRegression()\n",
        "        model.fit(X_train, y_train)\n",
        "\n",
        "        y_pred = model.predict(X_test)\n",
        "        acc = accuracy_score(y_test, y_pred)\n",
        "        print(f\"Accuracy on test set: {acc:.3f}\")\n",
        "\n",
        "        cm = confusion_matrix(y_test, y_pred)\n",
        "        print(\"Confusion matrix:\n",
        "\", cm)\n",
        "\n",
        "        plt.figure()\n",
        "        plt.bar([\"Correct\", \"Incorrect\"], [cm.trace(), cm.sum() - cm.trace()], color=[\"#4c72b0\", \"#dd8452\"])\n",
        "        plt.title(\"Prediction Outcomes\")\n",
        "        plt.ylabel(\"Number of samples\")\n",
        "        plt.show()\n",
        "\n",
        "        interpretation = \"\"  # Write one sentence about the accuracy meaning.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f766ac4d",
      "metadata": {},
      "source": [
        "\n",
        "## Check Your Understanding\n",
        "- Fill in `interpretation` above with a one-sentence takeaway.\n",
        "- Then run the checks below.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "9bb1c2d9",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "assert 'model' in globals(), \"Did you define the model variable?\"\n",
        "assert 'acc' in globals(), \"Did you compute accuracy and store it in acc?\"\n",
        "assert acc > 0, \"Accuracy should be positive. Did the training cell run?\"\n",
        "assert interpretation.strip() != \"\", \"Write a short interpretation sentence.\"\n",
        "print(\"Awesome! You trained a classifier and interpreted the metric.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "bd06d9d3",
      "metadata": {},
      "source": [
        "\n",
        "## Ethics & Impact Reflection\n",
        "- Where could bias show up in a dataset like this?\n",
        "- If accuracy is high but certain groups are misclassified, what should you do next?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "dfec698b",
      "metadata": {},
      "source": [
        "\n",
        "## Challenge / Extension (Optional)\n",
        "- Try a `DecisionTreeClassifier()` in place of logistic regression.\n",
        "- Compare accuracies and note which model handles any overlapping points better.\n",
        "- Add your findings to the summary.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "02e8a70e",
      "metadata": {},
      "source": [
        "\n",
        "## Summary & What's Next\n",
        "- You generated, split, and modeled a small dataset.\n",
        "- Accuracy and confusion matrices tell complementary stories.\n",
        "- Responsible AI asks: who benefits, and who might be harmed?\n",
        "\n",
        "**Next time:** We'll stress-test models under shifting data.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "cace1ce6",
      "metadata": {},
      "outputs": [],
      "source": [
        "if SHOW_TEACHER_KEY:\n",
        "    from IPython.display import Markdown\n",
        "    Markdown('\\n\"\"\"🗝️ Teacher Notes\"\"\"\\n- Typical accuracy: around 0.92–0.96 with given seed.\\n- Confusion matrix often near diagonal (e.g., [[36, 1], [3, 35]]).\\n- Encourage interpretations mentioning percent of correct predictions and need for fairness checks.\\n')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
''',
    'u01_l01_ethics_minilab_intro_ai_v1.ipynb': '''
{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "c2557348",
      "metadata": {},
      "source": [
        "# When Class Balance Changes: Accuracy Under Pressure\n",
        "_Lesson 01 – What is AI? Automation vs. Learning_\\n\\n**Authors:** LACES AIML\\n\\n**Standards Alignment**\\nCA CS Standards: 3A-IC-26; 3A-DA-09; 3B-AP-14\n",
        "LAUSD Digital Citizenship: Privacy & Security; Equity & Access (STEAM/CTE)\\n\\n**Time Estimate:** 25 minutes\\n\\n**Learning Objectives**\\n- Differentiate rule-based automation from machine learning with examples\n",
        "- Run a tiny ML pipeline and interpret a basic metric\n",
        "- Explain one ethical consideration (bias, privacy, or accountability) in context"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "1c2a9006",
      "metadata": {},
      "outputs": [],
      "source": [
        "# Toggle for teacher key\n",
        "SHOW_TEACHER_KEY = False\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.datasets import make_classification\n",
        "import random\n",
        "\n",
        "np.random.seed(42)\n",
        "random.seed(42)\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "\n",
        "%matplotlib inline\n",
        "plt.style.use('seaborn-v0_8-colorblind')\n",
        "plt.rcParams['figure.figsize'] = (6, 4)\n",
        "plt.rcParams['axes.titlesize'] = 12\n",
        "plt.rcParams['axes.labelsize'] = 11"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "80b9d176",
      "metadata": {},
      "source": [
        "## Concept Primer\n",
        "\n",
        "- **Scenario:** Screening scholarship applicants with AI.\n",
        "- **Class balance** (who gets labeled positive vs. negative) shapes metrics.\n",
        "- Monitoring shifts keeps decisions equitable.\n",
        "\n",
        "```\n",
        "[Collect Data] -> [Model trains] -> [Deploy] -> [Community Feedback]\n",
        "          ^                                  |\n",
        "          |----------------------------------|\n",
        "```\n",
        "\n",
        "We must re-evaluate models when the data mix changes.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "a2439bb0",
      "metadata": {},
      "source": [
        "## Guided Activity A\n",
        "### Scholarship Screening Scenario"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "e0742f05",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "X_balanced, y_balanced = make_classification(\n",
        "    n_samples=400,\n",
        "    n_features=3,\n",
        "    n_redundant=0,\n",
        "    n_clusters_per_class=1,\n",
        "    weights=[0.5, 0.5],\n",
        "    class_sep=1.2,\n",
        "    random_state=42,\n",
        ")\n",
        "\n",
        "model = LogisticRegression()\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced\n",
        ")\n",
        "model.fit(X_train, y_train)\n",
        "base_preds = model.predict(X_test)\n",
        "base_acc = accuracy_score(y_test, base_preds)\n",
        "print(f\"Balanced accuracy: {base_acc:.3f}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "abe4d9b3",
      "metadata": {},
      "source": [
        "\n",
        "### Predict Before You Run\n",
        "1. What does an accuracy near 0.5 mean in this context?\n",
        "2. Why might balanced data help fairness?\n",
        "3. What other metrics would you track?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6fba6887",
      "metadata": {},
      "source": [
        "## Guided Activity B\n",
        "### 🧪 Try It: Shift the Class Balance"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "fe9f2f0e",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "new_positive_weight = 0.2  # Adjust this to simulate fewer scholarships available.\n",
        "X_shifted, y_shifted = make_classification(\n",
        "    n_samples=400,\n",
        "    n_features=3,\n",
        "    n_redundant=0,\n",
        "    n_clusters_per_class=1,\n",
        "    weights=[1 - new_positive_weight, new_positive_weight],\n",
        "    class_sep=1.2,\n",
        "    random_state=24,\n",
        ")\n",
        "\n",
        "X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(\n",
        "    X_shifted, y_shifted, test_size=0.3, random_state=24, stratify=y_shifted\n",
        ")\n",
        "shift_model = LogisticRegression()\n",
        "shift_model.fit(X_train_s, y_train_s)\n",
        "shifted_preds = shift_model.predict(X_test_s)\n",
        "shifted_acc = accuracy_score(y_test_s, shifted_preds)\n",
        "\n",
        "print(f\"Shifted accuracy: {shifted_acc:.3f}\")\n",
        "\n",
        "base_report = classification_report(y_test, base_preds, output_dict=True)\n",
        "shifted_report = classification_report(y_test_s, shifted_preds, output_dict=True)\n",
        "\n",
        "metrics_df = pd.DataFrame({\n",
        "    'Scenario': ['Balanced', 'Shifted'],\n",
        "    'Accuracy': [base_acc, shifted_acc],\n",
        "    'Precision_Positive': [base_report['1']['precision'], shifted_report['1']['precision']],\n",
        "    'Recall_Positive': [base_report['1']['recall'], shifted_report['1']['recall']]\n",
        "})\n",
        "display(metrics_df)\n",
        "\n",
        "student_reflection = \"\"  # Write what changed and why it matters.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "4c898b0a",
      "metadata": {},
      "source": [
        "\n",
        "## Check Your Understanding\n",
        "- Complete `student_reflection` with 2–3 sentences.\n",
        "- Run the checks to confirm your observations.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "50cb35cc",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "assert 'shifted_acc' in globals(), \"Did you compute shifted_acc?\"\n",
        "assert isinstance(student_reflection, str) and len(student_reflection.strip()) >= 20, \"Write at least 20 characters reflecting on the shift.\"\n",
        "if shifted_acc >= base_acc:\n",
        "    print(\"Interesting! Sometimes accuracy rises even if positive class recall drops.\")\n",
        "else:\n",
        "    print(\"Accuracy dropped—check which class suffered in precision/recall.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "ba879c29",
      "metadata": {},
      "source": [
        "\n",
        "## Ethics & Impact Reflection\n",
        "- Who might be harmed if accuracy stays high but recall for scholarship winners drops?\n",
        "- How could you involve students and counselors in auditing this system?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "d9da0714",
      "metadata": {},
      "source": [
        "\n",
        "## Summary & What's Next\n",
        "- Accuracy alone can hide shifts when class balance changes.\n",
        "- Precision and recall reveal which group loses out.\n",
        "- Ethical reviews should happen whenever the data pipeline changes.\n",
        "\n",
        "**Next time:** Design safeguards and escalation paths for ML deployments.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "88b47dd6",
      "metadata": {},
      "outputs": [],
      "source": [
        "if SHOW_TEACHER_KEY:\n",
        "    from IPython.display import Markdown\n",
        "    Markdown('\\n\"\"\"🗝️ Teacher Notes\"\"\"\\n- Balanced accuracy typically ~0.87; shifted accuracy may remain ~0.90 while recall drops.\\n- Encourage students to mention how fewer positives reduce recall and fairness.\\n- Highlight the role of human oversight in scholarship decisions.\\n')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
''',
    'u01_l01_foundations_intro_ai_v1.ipynb': '''
{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "55d0bd70",
      "metadata": {},
      "source": [
        "# Automation vs Learning — Foundations\n",
        "_Lesson 01 – What is AI? Automation vs. Learning_\\n\\n**Authors:** LACES AIML\\n\\n**Standards Alignment**\\nCA CS Standards: 3A-IC-26; 3A-DA-09; 3B-AP-14\n",
        "LAUSD Digital Citizenship: Privacy & Security; Equity & Access (STEAM/CTE)\\n\\n**Time Estimate:** 25 minutes\\n\\n**Learning Objectives**\\n- Differentiate rule-based automation from machine learning with examples\n",
        "- Run a tiny ML pipeline and interpret a basic metric\n",
        "- Explain one ethical consideration (bias, privacy, or accountability) in context"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "eb69db03",
      "metadata": {},
      "outputs": [],
      "source": [
        "# Toggle for teacher key\n",
        "SHOW_TEACHER_KEY = False\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.datasets import make_classification\n",
        "import random\n",
        "\n",
        "np.random.seed(42)\n",
        "random.seed(42)\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "\n",
        "%matplotlib inline\n",
        "plt.style.use('seaborn-v0_8-colorblind')\n",
        "plt.rcParams['figure.figsize'] = (6, 4)\n",
        "plt.rcParams['axes.titlesize'] = 12\n",
        "plt.rcParams['axes.labelsize'] = 11"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "c25ca156",
      "metadata": {},
      "source": [
        "## Concept Primer\n",
        "\n",
        "- **Automation** follows instructions we explicitly program.\n",
        "- **Machine learning (ML)** finds patterns from data to make decisions.\n",
        "- Think of a vending machine: rules like `if money >= price`. An ML system learns which snack you might want next based on past choices.\n",
        "\n",
        "```\n",
        "[Sensors]-->[Rules?]-->[Action]\n",
        "     |           ^\n",
        "     v           |\n",
        "  [Data]----> [Model Learns]\n",
        "```\n",
        "\n",
        "**Key idea:** Automation is like a flowchart; ML is like a student practicing many examples until confident.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "7a22d37e",
      "metadata": {},
      "source": [
        "## Guided Activity A\n",
        "### Compare Automation and Learning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "bfd9f545",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "        print(\"Rule-based automation demo\")\n",
        "\n",
        "        def thermostat_rule(temp_f):\n",
        "            if temp_f < 68:\n",
        "                return \"Turn heater ON\"\n",
        "            elif temp_f > 75:\n",
        "                return \"Turn cooler ON\"\n",
        "            else:\n",
        "                return \"Stay steady\"\n",
        "\n",
        "        temps = [60, 70, 80]\n",
        "        for t in temps:\n",
        "            print(f\"At {t}°F -> {thermostat_rule(t)}\")\n",
        "\n",
        "        print(\"\n",
        "Learning-based demo (fitting from data)\")\n",
        "        study_hours = np.array([[1], [2], [3], [4], [5], [6]])\n",
        "        passed = np.array([0, 0, 0, 1, 1, 1])\n",
        "        model = LogisticRegression()\n",
        "        model.fit(study_hours, passed)\n",
        "\n",
        "        for h in [[2.5], [4.5]]:\n",
        "            pred = model.predict(h)[0]\n",
        "            prob = model.predict_proba(h)[0, 1]\n",
        "            print(f\"Study hours: {h[0]} -> Predicted pass? {bool(pred)} (confidence {prob:.2f})\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "059e2c93",
      "metadata": {},
      "source": [
        "\n",
        "### Predict Before You Run\n",
        "1. Which approach reacts instantly to new instructions?\n",
        "2. Which approach might improve with more data?\n",
        "3. What happens if the temperature rules see an unseen value like 90°F?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "9986b0d2",
      "metadata": {},
      "source": [
        "## Guided Activity B\n",
        "### 🧪 Try It: Edit the Rule"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "eede1b24",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "print(\"🧪 Try It: Update the rule thresholds and re-run!\")\n",
        "low_threshold = 65  # Try adjusting\n",
        "high_threshold = 78  # Try adjusting\n",
        "\n",
        "def custom_thermostat(temp_f):\n",
        "    if temp_f < low_threshold:\n",
        "        return \"Heat\"\n",
        "    elif temp_f > high_threshold:\n",
        "        return \"Cool\"\n",
        "    else:\n",
        "        return \"Comfort\"\n",
        "\n",
        "student_hours = np.linspace(1, 6, num=6).reshape(-1, 1)\n",
        "student_labels = np.array([0, 0, 0, 1, 1, 1])\n",
        "student_model = LogisticRegression()\n",
        "student_model.fit(student_hours, student_labels)\n",
        "\n",
        "temps_to_check = [62, 70, 79]\n",
        "for t in temps_to_check:\n",
        "    action = custom_thermostat(t)\n",
        "    print(f\"Temp {t}°F -> {action}\")\n",
        "\n",
        "new_hours = [[3.5], [5.5]]\n",
        "predictions = student_model.predict(new_hours)\n",
        "for h, p in zip(new_hours, predictions):\n",
        "    print(f\"Study hours {h[0]} -> predicted pass? {bool(p)}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f54d4d1f",
      "metadata": {},
      "source": [
        "## Check Your Understanding\n",
        "Answer the prompts below, then run the checks."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "ee497c3f",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "# 📝 Update these answers using complete sentences or keywords.\n",
        "answer_automation = \"\"\n",
        "answer_learning = \"\"\n",
        "answer_change = \"\"\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d71f6d04",
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "assert answer_automation.strip() != \"\", \"Describe automation in answer_automation.\"\n",
        "assert \"rule\" in answer_automation.lower(), \"Mention rules in answer_automation.\"\n",
        "assert answer_learning.strip() != \"\", \"Explain learning in answer_learning.\"\n",
        "assert any(word in answer_learning.lower() for word in [\"data\", \"examples\", \"patterns\"]), \"Mention data or patterns in answer_learning.\"\n",
        "assert answer_change.strip() != \"\", \"Explain what changes when data shifts in answer_change.\"\n",
        "print(\"Great! Your descriptions capture the core ideas.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f55a30b5",
      "metadata": {},
      "source": [
        "\n",
        "## Ethics & Impact Reflection\n",
        "- How might relying on a learned model be unfair if the training data ignores certain groups?\n",
        "- Where should humans stay in the loop to double-check automated decisions?\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "a6b668b2",
      "metadata": {},
      "source": [
        "\n",
        "## Challenge / Extension (Optional)\n",
        "- Change the dataset so that only students with 7+ hours are recorded.\n",
        "- Retrain the model and observe what happens to predictions around 4–5 hours.\n",
        "- Write down how the shift resembles or differs from rewriting the rule.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "1fc74171",
      "metadata": {},
      "source": [
        "\n",
        "## Summary & What's Next\n",
        "- Automation follows explicit instructions; learning adjusts to data.\n",
        "- Tiny models can learn thresholds but rely on good examples.\n",
        "- Ethical thinking keeps technology aligned with community values.\n",
        "\n",
        "**Next time:** We will build a full mini-classifier and measure its accuracy.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "112be600",
      "metadata": {},
      "outputs": [],
      "source": [
        "if SHOW_TEACHER_KEY:\n",
        "    from IPython.display import Markdown\n",
        "    Markdown('\\n\"\"\"🗝️ Teacher Notes\"\"\"\\n- Rule-based outputs: 60°F -> Turn heater ON, 80°F -> Turn cooler ON.\\n- Logistic regression predictions: around 2.5 hours -> False (low confidence), 4.5 hours -> True.\\n- Expected student reflections mention that data-driven approaches adapt but may inherit bias.\\n')"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
''',
}

def write_notebooks(output_dir: str = ".") -> None:
    """Write the embedded notebooks to *output_dir*."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for filename, payload in NOTEBOOK_JSON.items():
        target = out_path / filename
        target.write_text(payload, encoding="utf-8")
        print(f"Wrote {target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit embedded notebook JSON payloads.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where notebook files should be written (default: current directory)",
    )
    args = parser.parse_args()
    write_notebooks(args.output_dir)
