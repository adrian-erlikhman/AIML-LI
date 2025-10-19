# -*- coding: utf-8 -*-
import json
from types import SimpleNamespace

try:
    import nbformat as nbf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when nbformat isn't available
    def _ensure_list(source):
        if isinstance(source, list):
            return source
        if isinstance(source, str):
            return source.splitlines(True)
        raise TypeError("Cell source must be a string or list of strings")

    class _V4:
        @staticmethod
        def new_markdown_cell(source, metadata=None):
            return {
                "cell_type": "markdown",
                "metadata": metadata or {},
                "source": _ensure_list(source),
            }

        @staticmethod
        def new_code_cell(source, metadata=None):
            return {
                "cell_type": "code",
                "metadata": metadata or {},
                "execution_count": None,
                "outputs": [],
                "source": _ensure_list(source),
            }

        @staticmethod
        def new_notebook(cells=None, metadata=None):
            return {
                "cells": list(cells or []),
                "metadata": metadata or {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }

    def _write(nb_obj, filename):
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(nb_obj, handle, ensure_ascii=False, indent=2)

    nbf = SimpleNamespace(v4=_V4(), write=_write)
from textwrap import dedent

unit_code = "u01"
lesson_number = "l01"
short_keyword = "intro_ai"
standards = "CA CS Standards: 3A-IC-26; 3A-DA-09; 3B-AP-14\nLAUSD Digital Citizenship: Privacy & Security; Equity & Access (STEAM/CTE)"
objectives = [
    "Differentiate rule-based automation from machine learning with examples",
    "Run a tiny ML pipeline and interpret a basic metric",
    "Explain one ethical consideration (bias, privacy, or accountability) in context",
]


def make_setup_cell():
    base = "# Toggle for teacher key\nSHOW_TEACHER_KEY = False\n"
    setup = dedent(
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import make_classification
        import random

        np.random.seed(42)
        random.seed(42)
        try:
            import google.colab
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False

        %matplotlib inline
        plt.style.use('seaborn-v0_8-colorblind')
        plt.rcParams['figure.figsize'] = (6, 4)
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        """
    ).strip()
    return nbf.v4.new_code_cell(base + "\n" + setup)


def title_cell(notebook_title, time_estimate="25 minutes"):
    obj_lines = "\n".join([f"- {obj}" for obj in objectives])
    text = f"""# {notebook_title}\n_Lesson 01 – What is AI? Automation vs. Learning_\\n\\n**Authors:** LACES AIML\\n\\n**Standards Alignment**\\n{standards}\\n\\n**Time Estimate:** {time_estimate}\\n\\n**Learning Objectives**\\n{obj_lines}"""
    return nbf.v4.new_markdown_cell(text)


def concept_primer_cell(content):
    return nbf.v4.new_markdown_cell("## Concept Primer\n" + content)


def teacher_notes_cell(content):
    code = dedent(
        f"""
        if SHOW_TEACHER_KEY:
            from IPython.display import Markdown
            Markdown({content!r})
        """
    ).strip()
    return nbf.v4.new_code_cell(code)


def build_foundations():
    cells = [title_cell("Automation vs Learning — Foundations"), make_setup_cell()]
    concept_text = dedent(
        """
        - **Automation** follows instructions we explicitly program.\n        - **Machine learning (ML)** finds patterns from data to make decisions.\n        - Think of a vending machine: rules like `if money >= price`. An ML system learns which snack you might want next based on past choices.\n\n        ```
        [Sensors]-->[Rules?]-->[Action]
             |           ^
             v           |
          [Data]----> [Model Learns]
        ```\n\n        **Key idea:** Automation is like a flowchart; ML is like a student practicing many examples until confident.
        """
    )
    cells.append(concept_primer_cell(concept_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity A\n### Compare Automation and Learning"))
    activity_a_code = dedent(
        """
        print("Rule-based automation demo")

        def thermostat_rule(temp_f):
            if temp_f < 68:
                return "Turn heater ON"
            elif temp_f > 75:
                return "Turn cooler ON"
            else:
                return "Stay steady"

        temps = [60, 70, 80]
        for t in temps:
            print(f"At {t}°F -> {thermostat_rule(t)}")

        print("\nLearning-based demo (fitting from data)")
        study_hours = np.array([[1], [2], [3], [4], [5], [6]])
        passed = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression()
        model.fit(study_hours, passed)

        for h in [[2.5], [4.5]]:
            pred = model.predict(h)[0]
            prob = model.predict_proba(h)[0, 1]
            print(f"Study hours: {h[0]} -> Predicted pass? {bool(pred)} (confidence {prob:.2f})")
        """
    )
    cells.append(nbf.v4.new_code_cell(activity_a_code))

    predict_text = dedent(
        """
        ### Predict Before You Run
        1. Which approach reacts instantly to new instructions?
        2. Which approach might improve with more data?
        3. What happens if the temperature rules see an unseen value like 90°F?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(predict_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity B\n### 🧪 Try It: Edit the Rule"))
    try_it_code = dedent(
        """
        print("🧪 Try It: Update the rule thresholds and re-run!")
        low_threshold = 65  # Try adjusting
        high_threshold = 78  # Try adjusting

        def custom_thermostat(temp_f):
            if temp_f < low_threshold:
                return "Heat"
            elif temp_f > high_threshold:
                return "Cool"
            else:
                return "Comfort"

        student_hours = np.linspace(1, 6, num=6).reshape(-1, 1)
        student_labels = np.array([0, 0, 0, 1, 1, 1])
        student_model = LogisticRegression()
        student_model.fit(student_hours, student_labels)

        temps_to_check = [62, 70, 79]
        for t in temps_to_check:
            action = custom_thermostat(t)
            print(f"Temp {t}°F -> {action}")

        new_hours = [[3.5], [5.5]]
        predictions = student_model.predict(new_hours)
        for h, p in zip(new_hours, predictions):
            print(f"Study hours {h[0]} -> predicted pass? {bool(p)}")
        """
    )
    cells.append(nbf.v4.new_code_cell(try_it_code))

    check_intro = "## Check Your Understanding\nAnswer the prompts below, then run the checks."
    cells.append(nbf.v4.new_markdown_cell(check_intro))

    answers_code = dedent(
        """
        # 📝 Update these answers using complete sentences or keywords.
        answer_automation = ""
        answer_learning = ""
        answer_change = ""
        """
    )
    cells.append(nbf.v4.new_code_cell(answers_code))

    checks_code = dedent(
        """
        assert answer_automation.strip() != "", "Describe automation in answer_automation."
        assert "rule" in answer_automation.lower(), "Mention rules in answer_automation."
        assert answer_learning.strip() != "", "Explain learning in answer_learning."
        assert any(word in answer_learning.lower() for word in ["data", "examples", "patterns"]), "Mention data or patterns in answer_learning."
        assert answer_change.strip() != "", "Explain what changes when data shifts in answer_change."
        print("Great! Your descriptions capture the core ideas.")
        """
    )
    cells.append(nbf.v4.new_code_cell(checks_code))

    ethics_text = dedent(
        """
        ## Ethics & Impact Reflection
        - How might relying on a learned model be unfair if the training data ignores certain groups?
        - Where should humans stay in the loop to double-check automated decisions?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(ethics_text))

    challenge_text = dedent(
        """
        ## Challenge / Extension (Optional)
        - Change the dataset so that only students with 7+ hours are recorded.
        - Retrain the model and observe what happens to predictions around 4–5 hours.
        - Write down how the shift resembles or differs from rewriting the rule.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(challenge_text))

    summary_text = dedent(
        """
        ## Summary & What's Next
        - Automation follows explicit instructions; learning adjusts to data.
        - Tiny models can learn thresholds but rely on good examples.
        - Ethical thinking keeps technology aligned with community values.

        **Next time:** We will build a full mini-classifier and measure its accuracy.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(summary_text))

    teacher_notes = dedent(
        '''
        """\U0001F5DD\uFE0F Teacher Notes"""
        - Rule-based outputs: 60°F -> Turn heater ON, 80°F -> Turn cooler ON.
        - Logistic regression predictions: around 2.5 hours -> False (low confidence), 4.5 hours -> True.
        - Expected student reflections mention that data-driven approaches adapt but may inherit bias.
        '''
    )
    cells.append(teacher_notes_cell(teacher_notes))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
    return nb


def build_core():
    cells = [title_cell("Train Your First Classifier (Toy Data, Accuracy & Limits)"), make_setup_cell()]
    concept_text = dedent(
        """
        - **Pipeline:** collect data → split into train/test → train a model → evaluate.
        - **Accuracy** measures how often predictions match reality.
        - Small datasets help us see each step clearly before scaling up.

        ```
        [Data] -> [Split] -> [Model learns] -> [Metrics] -> [Decisions]
        ```

        Keep track of each stage to debug and explain choices.
        """
    )
    cells.append(concept_primer_cell(concept_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity A\n### Generate and Explore a Tiny Dataset"))
    dataset_code = dedent(
        """
        features, labels = make_classification(
            n_samples=300,
            n_features=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=42,
        )
        df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
        df["label"] = labels
        print(df.head())

        plt.figure()
        for label_value, marker in [(0, 'o'), (1, '^')]:
            subset = df[df['label'] == label_value]
            plt.scatter(subset['feature_1'], subset['feature_2'], label=f"Class {label_value}", alpha=0.7)
        plt.title("Tiny Classification Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
        """
    )
    cells.append(nbf.v4.new_code_cell(dataset_code))

    predict_text = dedent(
        """
        ### Predict Before You Run
        1. Are the two classes clearly separated?
        2. Which class seems larger?
        3. How might overlapping points affect accuracy?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(predict_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity B\n### 🧪 Try It: Train, Predict, Interpret"))
    guided_b_code = dedent(
        """
        X = df[["feature_1", "feature_2"]].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {acc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)

        plt.figure()
        plt.bar(["Correct", "Incorrect"], [cm.trace(), cm.sum() - cm.trace()], color=["#4c72b0", "#dd8452"])
        plt.title("Prediction Outcomes")
        plt.ylabel("Number of samples")
        plt.show()

        interpretation = ""  # Write one sentence about the accuracy meaning.
        """
    )
    cells.append(nbf.v4.new_code_cell(guided_b_code))

    check_intro = dedent(
        """
        ## Check Your Understanding
        - Fill in `interpretation` above with a one-sentence takeaway.
        - Then run the checks below.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(check_intro))

    checks_code = dedent(
        """
        assert 'model' in globals(), "Did you define the model variable?"
        assert 'acc' in globals(), "Did you compute accuracy and store it in acc?"
        assert acc > 0, "Accuracy should be positive. Did the training cell run?"
        assert interpretation.strip() != "", "Write a short interpretation sentence."
        print("Awesome! You trained a classifier and interpreted the metric.")
        """
    )
    cells.append(nbf.v4.new_code_cell(checks_code))

    ethics_text = dedent(
        """
        ## Ethics & Impact Reflection
        - Where could bias show up in a dataset like this?
        - If accuracy is high but certain groups are misclassified, what should you do next?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(ethics_text))

    challenge_text = dedent(
        """
        ## Challenge / Extension (Optional)
        - Try a `DecisionTreeClassifier()` in place of logistic regression.
        - Compare accuracies and note which model handles any overlapping points better.
        - Add your findings to the summary.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(challenge_text))

    summary_text = dedent(
        """
        ## Summary & What's Next
        - You generated, split, and modeled a small dataset.
        - Accuracy and confusion matrices tell complementary stories.
        - Responsible AI asks: who benefits, and who might be harmed?

        **Next time:** We'll stress-test models under shifting data.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(summary_text))

    teacher_notes = dedent(
        '''
        """\U0001F5DD\uFE0F Teacher Notes"""
        - Typical accuracy: around 0.92–0.96 with given seed.
        - Confusion matrix often near diagonal (e.g., [[36, 1], [3, 35]]).
        - Encourage interpretations mentioning percent of correct predictions and need for fairness checks.
        '''
    )
    cells.append(teacher_notes_cell(teacher_notes))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
    return nb


def build_ethics():
    cells = [title_cell("When Class Balance Changes: Accuracy Under Pressure"), make_setup_cell()]
    concept_text = dedent(
        """
        - **Scenario:** Screening scholarship applicants with AI.
        - **Class balance** (who gets labeled positive vs. negative) shapes metrics.
        - Monitoring shifts keeps decisions equitable.

        ```
        [Collect Data] -> [Model trains] -> [Deploy] -> [Community Feedback]
                  ^                                  |
                  |----------------------------------|
        ```

        We must re-evaluate models when the data mix changes.
        """
    )
    cells.append(concept_primer_cell(concept_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity A\n### Scholarship Screening Scenario"))
    guided_a_code = dedent(
        """
        X_balanced, y_balanced = make_classification(
            n_samples=400,
            n_features=3,
            n_redundant=0,
            n_clusters_per_class=1,
            weights=[0.5, 0.5],
            class_sep=1.2,
            random_state=42,
        )

        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        model.fit(X_train, y_train)
        base_preds = model.predict(X_test)
        base_acc = accuracy_score(y_test, base_preds)
        print(f"Balanced accuracy: {base_acc:.3f}")
        """
    )
    cells.append(nbf.v4.new_code_cell(guided_a_code))

    predict_text = dedent(
        """
        ### Predict Before You Run
        1. What does an accuracy near 0.5 mean in this context?
        2. Why might balanced data help fairness?
        3. What other metrics would you track?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(predict_text))

    cells.append(nbf.v4.new_markdown_cell("## Guided Activity B\n### 🧪 Try It: Shift the Class Balance"))
    guided_b_code = dedent(
        """
        new_positive_weight = 0.2  # Adjust this to simulate fewer scholarships available.
        X_shifted, y_shifted = make_classification(
            n_samples=400,
            n_features=3,
            n_redundant=0,
            n_clusters_per_class=1,
            weights=[1 - new_positive_weight, new_positive_weight],
            class_sep=1.2,
            random_state=24,
        )

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_shifted, y_shifted, test_size=0.3, random_state=24, stratify=y_shifted
        )
        shift_model = LogisticRegression()
        shift_model.fit(X_train_s, y_train_s)
        shifted_preds = shift_model.predict(X_test_s)
        shifted_acc = accuracy_score(y_test_s, shifted_preds)

        print(f"Shifted accuracy: {shifted_acc:.3f}")

        base_report = classification_report(y_test, base_preds, output_dict=True)
        shifted_report = classification_report(y_test_s, shifted_preds, output_dict=True)

        metrics_df = pd.DataFrame({
            'Scenario': ['Balanced', 'Shifted'],
            'Accuracy': [base_acc, shifted_acc],
            'Precision_Positive': [base_report['1']['precision'], shifted_report['1']['precision']],
            'Recall_Positive': [base_report['1']['recall'], shifted_report['1']['recall']]
        })
        display(metrics_df)

        student_reflection = ""  # Write what changed and why it matters.
        """
    )
    cells.append(nbf.v4.new_code_cell(guided_b_code))

    check_intro = dedent(
        """
        ## Check Your Understanding
        - Complete `student_reflection` with 2–3 sentences.
        - Run the checks to confirm your observations.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(check_intro))

    checks_code = dedent(
        """
        assert 'shifted_acc' in globals(), "Did you compute shifted_acc?"
        assert isinstance(student_reflection, str) and len(student_reflection.strip()) >= 20, "Write at least 20 characters reflecting on the shift."
        if shifted_acc >= base_acc:
            print("Interesting! Sometimes accuracy rises even if positive class recall drops.")
        else:
            print("Accuracy dropped—check which class suffered in precision/recall.")
        """
    )
    cells.append(nbf.v4.new_code_cell(checks_code))

    ethics_text = dedent(
        """
        ## Ethics & Impact Reflection
        - Who might be harmed if accuracy stays high but recall for scholarship winners drops?
        - How could you involve students and counselors in auditing this system?
        """
    )
    cells.append(nbf.v4.new_markdown_cell(ethics_text))

    summary_text = dedent(
        """
        ## Summary & What's Next
        - Accuracy alone can hide shifts when class balance changes.
        - Precision and recall reveal which group loses out.
        - Ethical reviews should happen whenever the data pipeline changes.

        **Next time:** Design safeguards and escalation paths for ML deployments.
        """
    )
    cells.append(nbf.v4.new_markdown_cell(summary_text))

    teacher_notes = dedent(
        '''
        """\U0001F5DD\uFE0F Teacher Notes"""
        - Balanced accuracy typically ~0.87; shifted accuracy may remain ~0.90 while recall drops.
        - Encourage students to mention how fewer positives reduce recall and fairness.
        - Highlight the role of human oversight in scholarship decisions.
        '''
    )
    cells.append(teacher_notes_cell(teacher_notes))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
    return nb


if __name__ == "__main__":
    foundations_nb = build_foundations()
    core_nb = build_core()
    ethics_nb = build_ethics()

    nbf.write(foundations_nb, f"{unit_code}_{lesson_number}_foundations_{short_keyword}_v1.ipynb")
    nbf.write(core_nb, f"{unit_code}_{lesson_number}_core_{short_keyword}_v1.ipynb")
    nbf.write(ethics_nb, f"{unit_code}_{lesson_number}_ethics_minilab_{short_keyword}_v1.ipynb")
