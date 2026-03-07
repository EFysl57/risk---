class RiskModel:
    # You must set the `new_premiums` here
    def update_premiums(self):
        self.new_premiums = self.old_premiums
    
    def create(self, premiums, premiums_wo_term, claims_amount, claims_cnt):
        self.old_premiums = premiums
        self.old_premiums_wo_term = premiums_wo_term
        self.claims_amount = claims_amount
        self.claims_cnt = claims_cnt
        self.premiums_percent = premiums_wo_term / premiums

    def create_from_dataset(self, data):
        self.create(data['premium'], data['premium_wo_term'], 
                         data['claim_amount'], data['claim_cnt'])

    def update(self):
        self.update_premiums()
        # IMPORTANT: Clip new_premiums
        self.new_premiums.clip(0, self.old_premiums * 3, inplace=True) 
        self.new_premiums_wo_term = self.new_premiums * self.premiums_percent

    def all_stats(self):
        stats = self.balance_stats()
        stats["new_loss_ratio"] = self.new_loss_ratio()
        stats["old_loss_ratio"] = self.old_loss_ratio()
        return stats
    
    def balance_stats(self):
        increased = self.new_premiums > self.old_premiums
        not_increased = ~increased
        have_claims = self.claims_cnt > 0

        n_total = len(self.old_premiums)
        n_increased = increased.sum()
        n_not_increased = not_increased.sum()
        
        false_increase = increased & (~have_claims)
        false_decrease = (~increased) & have_claims
        n_false_inc = false_increase.sum()
        n_false_dec = false_decrease.sum()
    
        return {
            "increased": n_increased / n_total,
            "false_increase": n_false_inc / n_increased if n_increased > 0 else 0,
            "false_decrease": n_false_dec / n_not_increased if n_not_increased > 0 else 0
        }

    # Loss Ratio Calculation (Коэффициент выплат)
    def new_loss_ratio(self):
        return loss_ratio(self.claims_amount, self.new_premiums_wo_term)

    def old_loss_ratio(self):
        return loss_ratio(self.claims_amount, self.old_premiums_wo_term)


def loss_ratio(claims, premiums):
    total_claims = claims.sum()
    total_premium = premiums.sum()
    return (total_claims / total_premium) if total_premium != 0 else 0