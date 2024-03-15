import { createMuiTheme } from '@material-ui/core/styles';

export default createMuiTheme({
    overrides: {
      MUIDataTableHeadCell: {
        fixedHeader: {
          position:'static',
          backgroundColor: "#2a3f54e8",
          fontSize: "1rem",
          fontWeight: "600",
          color: "#ffffff",
        }
      },
      MuiPaper: {
        elevation4: {
          boxShadow: "none",
        }
      },
      MuiTableCell:{
        body:{
          fontSize: "0.95rem",
          cursor: "pointer",
        }
      },
      MuiTableRow: {
        head: {
          height: "45px",
        },
        root:{
          color: "inherit",
          height: "40px",
          display: "table-row",
          outline: "none",
          verticalAlign: "middle",
  
        }
      },
      MuiTypography: {
        h6: {
          color: "rgba(0, 0, 0, 0.87)",
          fontSize: "2.2rem",
          fontFamily: '"Helvetica Neue",Roboto,Arial,"Droid Sans",sans-serif',
          fontWeight: "500",
          lineHeight: "1.6",
          letterSpacing: "0.0075em",
        },
        caption: {
          color: "rgba(21, 21, 21, 0.87)",
          fontSize: "1rem",
          fontWeight: "600",
          fontFamily: '"Helvetica Neue",Roboto,Arial,"Droid Sans",sans-serif',
          lineHeight: "1.375em",
        }
      },
      MuiMenuItem: {
        root: {
          color: "rgba(0, 0, 0, 0.87)",
          width: "auto",
          height: "24px",
          overflow: "hidden",
          fontSize: "1rem",
          boxSizing: "content-box",
          fontWeight: "400",
          fontFamily: '"Helvetica Neue",Roboto,Arial,"Droid Sans",sans-serif',
          lineHeight: "1.5em",
          whiteSpace: "nowrap",
        }
      },
      MuiList: {
        padding: {
          paddingTop: "0px",
          paddingBottom: "0px",
        }
      },
      MuiToolbar: {
        regular: {
          paddingLeft: "0px",
          minHeight: "40px",  
          '@media (min-width: 600px)': {
               minHeight: "48px"
           },
           '@media (min-width: 0px) and (orientation: landscape)': {
               minHeight: "48px",
           }
        }
      },
      MuiIconButton: {
        root: {
            flex: "0 0 auto",
            color: "rgba(0, 0, 0, 0.54)",
            padding: "5px",
            overflow: "visible",
            fontSize: "1.5rem",
            textAlign: "center",
            transition: "background-color 150ms cubic-bezier(0.4, 0, 0.2, 1) 0ms",
            borderRadius: "50%",
        }
      },
      MuiTablePagination: {
        select: {
          paddingLeft: "8px",
          paddingRight: "20px",
          paddingTop: "10px",
          fontSize: "10px",
        }
      }
    }
})