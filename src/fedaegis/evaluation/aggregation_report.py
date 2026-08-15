import pandas as pd


class AggregationReport:

    def __init__(self):

        self.rows = []

    def add(

            self,

            report):

        self.rows.append({

            "client": report.client_id,

            "samples": report.samples,

            "fnr": report.fnr,

            "class_balance": report.class_balance,

            "reliability": report.reliability

        })

    def save(

            self,

            filename):

        pd.DataFrame(

            self.rows

        ).to_csv(

            filename,

            index=False

        )
