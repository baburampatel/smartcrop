# Consent Template — Crop Recommendation Field Trial

## Informed Consent Form

### Study Title
Evaluating an AI-Based Crop Recommendation System for Indian Agriculture

### Investigators
[Organization Name]
[Principal Investigator Name and Contact]

---

### Purpose
You are being invited to participate in a study that tests a new computer-based system for recommending crops. The system uses your soil test results to suggest which crops may grow best on your land.

### What will happen
1. A soil sample will be collected from your field (free of charge)
2. The soil will be tested at a government-approved laboratory
3. Depending on your group assignment, you may receive crop recommendations from our computer system
4. We will visit your field 2–3 times during the growing season
5. At harvest, we will record your crop yield
6. After harvest, we will ask you about your experience (survey)

### Your Rights
- **Voluntary**: You do not have to participate. You can stop at any time without penalty
- **Your choice**: Even if the system recommends a crop, **you decide what to plant**. The recommendation is advisory only
- **Free soil test**: Your soil test is provided at no cost regardless of participation
- **Privacy**: Your personal information will be kept confidential. Your name will be replaced with a code number in all records
- **No risk**: This study involves no physical risk. Following a crop recommendation is your personal choice

### Data We Collect

| Data | Purpose | Who sees it |
|------|---------|-------------|
| Name, phone | Contact only | Study team only (deleted after study) |
| Village, district | Location grouping | Anonymized in reports |
| Farm size | Research analysis | Anonymized in reports |
| Soil test results | Input to system | Anonymized in reports |
| Crop choice | Research outcome | Anonymized in reports |
| Yield | Research outcome | Anonymized in reports |

### Data Protection
- All personal data will be **stored securely** with encryption
- Your name will **never appear** in any publication or report
- Data will be **deleted within 3 years** after study completion
- Compliant with Indian data protection regulations

### Compensation
- Free soil testing (worth approximately ₹500)
- No other monetary compensation

### Questions
If you have questions about this study, contact:
- [PI Name] at [Phone/Email]
- [Ethics Committee Name] at [Phone/Email]

---

## Consent Declaration

I have read (or had read to me) the above information. I understand:

- [ ] My participation is voluntary
- [ ] I can withdraw at any time
- [ ] My data will be kept confidential
- [ ] The crop recommendation is advisory; I make the final decision
- [ ] I agree to soil sampling and data collection as described

**Participant:**

Name: ___________________________

Signature / Thumb impression: ___________________________

Date: ___________________________

Village: ___________________________

**Witness (if participant is non-literate):**

Name: ___________________________

Signature: ___________________________

Date: ___________________________

**Field Officer:**

Name: ___________________________

Signature: ___________________________

Date: ___________________________

---

## PII Redaction Process

All collected personal data undergoes the following redaction pipeline:

1. **At collection**: Assign unique `participant_id` (e.g., P001, P002)
2. **Linking table**: Name ↔ participant_id mapping stored in **separate encrypted database** with restricted access
3. **Analysis dataset**: Contains only participant_id, no names, phone numbers, or exact addresses
4. **Geolocation**: Coordinates rounded to 5km grid (no exact farm location)
5. **Publication**: Only aggregate statistics reported; no individual-level data
6. **Data retention**: Linking table deleted 3 years after study completion
7. **Audit trail**: All data access logged

### Risk Assessment for Misuse

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Incorrect recommendation causes crop failure | Medium | Low | Advisory only + expert override |
| Recommendation used for market manipulation | Low | Very Low | Individual-level, not aggregate market |
| Privacy breach of farmer data | Medium | Low | Encryption, minimal PII, access controls |
| Bias against certain crops/regions | Medium | Medium | Regular bias audits, diverse training data |
| Over-reliance on AI replacing expert knowledge | Medium | Medium | Training field officers, clear advisory framing |
