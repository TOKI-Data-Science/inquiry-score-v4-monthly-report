drop table t_inq_v4_bnpl_aged_pool;
drop table t_inq_v4_lease_aged_pool;
drop table t_inq_v4_credit_aged_pool;
drop table t_inq_v4_aged_pool;

create table t_inq_v4_bnpl_aged_pool as
with user_pool as (
    select distinct a.*, to_number(to_char(trunc(decision_engine_success_date), 'yyyymm')) as base_month, c.user_id
    from (select * from t_v4_report_202410_202512 where product_type in ('bnpl')) a
    inner join toki.bnpl_limit_request b on a.id = b.id_
    inner join toki.bnpl_account c on b.bnpl_account_id = c.id_
    inner join (
      select user_id, created_date as limit_calculation_date from toki.bnpl_account_history
      where created_by_action in ('limit_calculated')
        and bnpl_limit > 0
    ) d on c.user_id = d.user_id and to_number(to_char(trunc(decision_engine_success_date), 'yyyymmdd')) = to_number(to_char(trunc(d.limit_calculation_date), 'yyyymmdd'))
),
invoice_raw as (
  select distinct
    a.id_ as loan_request_id,
    to_number(to_char(trunc(a.created_at), 'yyyymmdd')) as loan_request_date,
    to_number(substr(to_char(trunc(a.created_at), 'yyyymmdd'), 1, 6)) as loan_request_month,
    a.amount as loan_request_amt,
    a.transaction_id as loan_transaction_id,
    a.account_id as user_id,
    a.bnpl_type as loan_type,
    a.method as loan_method,
    a.status as loan_status,
    b.id_ as invoice_id,
    b.amount as invoice_amt,
    to_number(to_char(trunc(b.payment_date), 'yyyymmdd')) as invoice_date,
    b.status as invoice_status,
    c.amount as repayment_amt,
    to_number(to_char(trunc(c.transaction_date), 'yyyymmdd')) as repayment_date,
    c.payment_type as repayment_type,
    c.status as repayment_status,
    b.payment_date as invoice_payment_date,
    c.transaction_date as repayment_transaction_date
  from toki.dpr_tajet_bnpl_request a
  inner join toki.dpr_tajet_bnpl_invoice b on a.transaction_id = b.transaction_id and a.status not in ('CANCELLED', 'PENDING')
  inner join toki.dpr_tajet_bnpl_repayment c on b.id_ = c.invoice_id and c.status in ('SUCCESS') and c.payment_type not in ('REFUND')
),
repayment_aggregation as (
  select
    i.loan_request_id,
    i.invoice_id,
    i.loan_type,
    i.loan_request_amt,
    i.invoice_amt,
    m.base_month,
    m.id,
    sum(case when i.loan_type = 'UNSTRICTED_1' and i.repayment_date <= to_number(to_char(last_day(to_date(to_char(m.base_month), 'yyyymm')), 'yyyymmdd')) then i.repayment_amt else 0 end) as total_repayment_model_unrestricted,
    max(case when i.loan_type = 'STRICTED_4' and i.repayment_date <= to_number(to_char(last_day(to_date(to_char(m.base_month), 'yyyymm')), 'yyyymmdd')) then i.repayment_amt else 0 end) as repayment_amt_model_restricted,
    sum(case when i.loan_type = 'UNSTRICTED_1' and i.repayment_date < to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), 'yyyymmdd')) then i.repayment_amt else 0 end) as total_repayment_before_base_unrestricted,
    max(case when i.loan_type = 'STRICTED_4' and i.repayment_date < to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), 'yyyymmdd')) then i.repayment_amt else 0 end) as repayment_amt_restricted
  from invoice_raw i
  right join user_pool m on i.user_id = m.user_id
  where to_number(substr(to_char(i.invoice_date), 1, 6))
    between to_number(to_char(add_months(to_date(to_char(m.base_month), 'yyyymm'), 0), 'yyyymm'))
        and to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), 'yyyymm'))
  group by i.loan_request_id, i.invoice_id, i.loan_type, i.loan_request_amt, i.invoice_amt, m.base_month, m.id
),
calculated_od as (
  select
    i.user_id,
    i.loan_request_id,
    i.invoice_id,
    i.loan_type,
    i.invoice_amt,
    i.loan_request_amt,
    i.repayment_type,
    i.repayment_transaction_date,
    i.invoice_payment_date,
    m.id,
    m.product_type,
    m.model_type,
    m.score_tag,
    m.v4_score,
    m.v4_bin,
    m.base_month,
    r.total_repayment_before_base_unrestricted,
    r.repayment_amt_restricted,
    case
      when i.loan_type = 'UNSTRICTED_1' and r.total_repayment_before_base_unrestricted >= i.loan_request_amt then 1
      when i.loan_type = 'STRICTED_4' and r.repayment_amt_restricted >= i.invoice_amt then 1
      else 0
    end as is_paid,
    case
      when to_number(substr(to_char(i.invoice_date), 1, 6)) between
           to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 1), 'yyyymm')) and
           to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), 'yyyymm'))
      then
        case
          when trunc(i.invoice_payment_date) > least(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), trunc(sysdate))
            then null
          when i.repayment_type in ('REPAYMENT')
            and (case
              when i.loan_type = 'UNSTRICTED_1' and r.total_repayment_before_base_unrestricted >= i.loan_request_amt then 1
              when i.loan_type = 'STRICTED_4' and r.repayment_amt_restricted >= i.invoice_amt then 1
              else 0
            end) = 1
            and i.repayment_transaction_date is not null
            and i.repayment_transaction_date <= add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12)
            then case when trunc(i.repayment_transaction_date) - trunc(i.invoice_payment_date) >= 0
                      then trunc(i.repayment_transaction_date) - trunc(i.invoice_payment_date) else 0 end
          when i.repayment_type in ('REPAYMENT')
            and (case
              when i.loan_type = 'UNSTRICTED_1' and r.total_repayment_before_base_unrestricted >= i.loan_request_amt then 1
              when i.loan_type = 'STRICTED_4' and r.repayment_amt_restricted >= i.invoice_amt then 1
              else 0
            end) = 0
            and i.repayment_transaction_date is not null
            then case when least(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.invoice_payment_date) >= 0
                      then least(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.invoice_payment_date) else 0 end
          when i.repayment_type is null
            then case when least(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.invoice_payment_date) >= 0
                      then least(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.invoice_payment_date) else 0 end
          else 0
        end
      else null
    end as target_od_raw
  from invoice_raw i
  right join user_pool m on i.user_id = m.user_id
  left join repayment_aggregation r on i.loan_request_id = r.loan_request_id and i.invoice_id = r.invoice_id and m.base_month = r.base_month and m.id = r.id
  where to_number(substr(to_char(i.invoice_date), 1, 6))
    between to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 1), 'yyyymm'))
        and to_number(to_char(add_months(last_day(to_date(to_char(m.base_month), 'yyyymm')), 12), 'yyyymm'))
)
select distinct
  id,
  product_type,
  model_type,
  score_tag,
  v4_score,
  v4_bin,
  base_month,
  user_id,
  max(case when target_od_raw >= 0 then target_od_raw else 0 end) as od,
  max(case when target_od_raw >= 90 then 1 else 0 end) as event
from calculated_od
group by id, product_type, model_type, score_tag, v4_score, v4_bin, user_id, base_month;

create table t_inq_v4_credit_aged_pool as
with user_pool as (
    select distinct a.*, to_number(to_char(trunc(decision_engine_is_successful_date), 'yyyymm')) as base_month, c.user_id
    from (select * from t_v4_report_202410_202512 where product_type in ('credit')) a
    inner join toki.credit_zms_request b on a.id = b.zms_request_id
    inner join toki.credit_credit c on b.borrower_id = c.borrower_id
    inner join (
      select credit_id, created_date as limit_calculation_date from toki.credit_credit_history
      where created_by_action in ('contract_accepted')
        and credit_limit > 0
    ) d on c.credit_id = d.credit_id and to_number(to_char(trunc(decision_engine_is_successful_date), 'yyyymmdd')) = to_number(to_char(trunc(d.limit_calculation_date), 'yyyymmdd'))
),
tmp_credit_invoice as (
  select
    b.id,
    b.product_type,
    b.model_type,
    b.score_tag,
    b.v4_score,
    b.v4_bin,
    b.base_month,
    b.user_id,
    i.invoice_id,
    i.invoice_type,
    i.principal_amt     as invoice_amt,
    i.target_month_date as invoice_month,
    i.fully_paid_date,
    i.due_date,
    case
      when trunc(i.due_date) between add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 1) and add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12)
      then
        case
          when trunc(i.due_date) > least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate))
            then null
          when i.fully_paid_date is not null
            and i.fully_paid_date <= least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate))
            then case when round(i.fully_paid_date - i.due_date, 0) >= 0
                      then round(i.fully_paid_date - i.due_date, 0) else 0 end
          else case when round(least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate)) - i.due_date, 0) >= 0
                    then round(least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate)) - i.due_date, 0) else 0 end
        end
      else null
    end as target_od_raw
  from toki.credit_invoice i
  inner join toki.credit_credit cc on i.credit_id = cc.credit_id   
  right join user_pool b on cc.user_id = b.user_id
    and trunc(i.due_date) between add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 1)
                              and add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12)
    and i.invoice_type = 'MONTHLY'
)
select
  id,
  product_type,
  model_type,
  score_tag,
  v4_score,
  v4_bin,
  base_month,
  user_id,
  max(case when target_od_raw >= 0 then target_od_raw else 0 end) as od,
  max(case when target_od_raw >= 90 then 1 else 0 end) as event
from tmp_credit_invoice
group by id, product_type, model_type, score_tag, v4_score, v4_bin, base_month, user_id;

create table t_inq_v4_lease_aged_pool as
with user_pool as (
    select distinct a.*, to_number(to_char(trunc(decision_engine_successful_date), 'yyyymm')) as base_month, b.user_id
    from (select * from t_v4_report_202410_202512 where product_type in ('leasing')) a
    inner join toki.handset_tmp_limit_request b on a.id = b.id
    inner join toki.handset_borrower c on b.user_id = c.user_id
    inner join toki.handset_loan d on c.id = d.borrower_id and to_number(to_char(trunc(decision_engine_successful_date), 'yyyymmdd')) = to_number(to_char(trunc(loan_activated_date), 'yyyymmdd'))
),
tmp_handset_invoice as (
    select
        b.loan_id,
        a.id as invoice_id,
        b.id as loan_invoice_id,
        a.invoice_type,
        to_date(a.due_date, 'dd-mon-yy') as due_date
    from toki.handset_invoice a
    left join toki.handset_loan_invoice b on a.id = b.invoice_id
),
tmp_handset_repayment as (
    select
        loan_id,
        loan_invoice_id,
        max(createdat) as createdat
    from (
        select
            loan_id,
            loan_invoice_id,
            to_date(created_date, 'dd-mon-yy') as createdat
        from toki.handset_loan_repayment
    )
    group by loan_id, loan_invoice_id
),
tmp_lease_invoice as (
  select
    b.id,
    b.product_type,
    b.model_type,
    b.score_tag,
    b.v4_score,
    b.v4_bin,
    b.base_month,
    b.user_id,
    i.invoice_id,
    i.invoice_type,
    i.due_date,
    rep.createdat as paid_date,
    case
      when trunc(i.due_date) between add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 1) and add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12)
      then
        case
          when trunc(i.due_date) > least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate))
            then null
          when rep.createdat is not null
            and rep.createdat <= least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate))
            then case when round(rep.createdat - i.due_date, 0) >= 0
                      then round(rep.createdat - i.due_date, 0) else 0 end
          else case when round(least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.due_date), 0) >= 0
                    then round(least(add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12), trunc(sysdate)) - trunc(i.due_date), 0) else 0 end
        end
      else null
    end as target_od_raw
  from tmp_handset_invoice i
  left join tmp_handset_repayment rep on i.loan_id = rep.loan_id and i.loan_invoice_id = rep.loan_invoice_id
  inner join toki.handset_loan o on i.loan_id = o.id
  inner join toki.handset_borrower br on o.borrower_id = br.id
  right join user_pool b on br.user_id = b.user_id
    and trunc(i.due_date) between add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 1)
                              and add_months(last_day(to_date(to_char(b.base_month), 'yyyymm')), 12)
  where i.invoice_type = 'SCHEDULED'
)
select
  id,
  product_type,
  model_type,
  score_tag,
  v4_score,
  v4_bin,
  base_month,
  user_id,
  max(case when target_od_raw >= 0 then target_od_raw else 0 end) as od,
  max(case when target_od_raw >= 90 then 1 else 0 end) as event
from tmp_lease_invoice
group by id, product_type, model_type, score_tag, v4_score, v4_bin, base_month, user_id;

create table t_inq_v4_aged_pool as 
select p.*
from (
  select id, product_type, model_type, score_tag, v4_score, v4_bin, base_month,
         to_char(user_id) as user_id, event
  from t_inq_v4_bnpl_aged_pool
  union
  select id, product_type, model_type, score_tag, v4_score, v4_bin, base_month,
         to_char(user_id) as user_id, event
  from t_inq_v4_lease_aged_pool
  union
  select id, product_type, model_type, score_tag, v4_score, v4_bin, base_month,
         to_char(user_id) as user_id, event
  from t_inq_v4_credit_aged_pool
) p
where p.base_month < to_number(to_char(add_months(trunc(sysdate), -12), 'yyyymm'))
  and p.base_month >= 202504;

