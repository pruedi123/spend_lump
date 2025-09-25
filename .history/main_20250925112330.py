if rows2:
    st.subheader("Median Withdrawal — Lump Sum")
    out = pd.DataFrame(rows2)
    fmt = out.copy()
    fmt["Annual Retirement Income (Historically)"] = fmt["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
    # Create formatted lifetime income column and drop the original numeric column
    fmt["Total Median Lifetime Retirement Income"] = fmt["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
    fmt = fmt.drop(columns=["Total Median Retirement Income"])
    # Optional: enforce column order
    fmt = fmt[[
        "Portfolio",
        "Years",
        "Annual Retirement Income (Historically)",
        "Total Median Lifetime Retirement Income",
    ]]
    st.dataframe(fmt, use_container_width=True)