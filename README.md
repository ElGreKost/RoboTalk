Installation Steps
1. Κατεβάζουμε Webots  (https://cyberbotics.com)
2. Κατεβάζουμε το repo:
   a)Κάνουμε κλικ στο πράσινο κουμπί <>code
   b)Επιλέγουμε download zip
3. Κάνουμε unzip
4. Μέσω terminal, μεταφερόμαστε στον φάκελο "deepbots" που μόλις κατεβάσαμε (πχ. "cd ~/Downloads/Robotalk-main/deepbots")
5. Εκτελούμε την εντολή "pip install ."

ΠΡΟΣΟΧΗ: Για να λειτουργήσει με επιτυχία η εγκατάσταση πρέπει η python να είναι σε version <= 3.11
Για να δείτε την έκδοση τρέχετε την παρακάτω εντολή στο terminal
```python3 --version```

Controllers που θα χρησιμοποιήσουμε:
- ppo_controller       : Διακριτά actions με την χρήση του PPO Agent
- continuous_controller: Συνεχή actions με την χρήση του PPO Agent
- imitation-robot      : Εκπαιδευμένο μοντέλο με την χρήση μεθόδων imitation
