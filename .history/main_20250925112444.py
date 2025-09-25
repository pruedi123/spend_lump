if rows2:
    st.subheader("Median Withdrawal — Lump Sum")
    out = pd.DataFrame(rows2)
    fmt = out.copy()
    # Format money columns with commas and no decimals
    fmt["Annual Retirement Income (Historically)"] = fmt["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
    fmt["Total Median Lifetime Retirement Income"] = fmt["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
    # Drop the original numeric lifetime column to avoid duplication
    fmt = fmt.drop(columns=["Total Median Retirement Income"], errors="ignore")
    # Enforce a tidy column order
    fmt = fmt[[
        "Portfolio",
        "Years",
        "Annual Retirement Income (Historically)",
        "Total Median Lifetime Retirement Income",
    ]]
    st.dataframe(fmt, width='stretch')