import { createMuiTheme } from '@material-ui/core/styles';
import color from './Constants/color'

export default createMuiTheme({
  overrides: {

    MUIDataTable: {
      // responsiveScroll: {
      //   maxHeight: '287px !important',
      // },
      responsiveScroll: {
        overflowX: 'none',
        height: 'auto',
        maxHeight: '287px !important',
      },
    },

    MUIDataTableSelectCell: {
      fixedHeader: {
        position: 'relative'
      },
      headerCell: {
        position: 'sticky'
      }
    },
    MUIDataTableHeadCell: {
      fixedHeader: {
        backgroundColor: color.primaryColor,
        fontSize: "1.2rem",
        fontWeight: 600,
        color: "#ffffff",
        position: 'static'
      }
    },
    MuiPaper: {
      elevation4: {
        boxShadow: "none",
      }
    },
    MuiTableCell: {
      body: {
        fontSize: "1.1rem",
        cursor: "pointer",
      },
      root: {
        display: "table-cell",
        padding: "4px 24px 4px 24px",
        textAlign: "left",
        borderBbottom: "1px solid rgba(224, 224, 224, 1)",
        verticalAlign: "inherit",
      }
    },
    MuiTableRow: {
      head: {
        height: "45px",
      },
      root: {
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
      }
    },

    // MUIDataTableResponsiveScroll4: {
    //     height: "100%",
    //     overflow: "auto",
    //     maxHeight: "287px",
    //     overflowX: "auto",
    // },

    MUIDataTableResponsiveScroll: {
      height: "100% !important",
      overflow: "auto !important",
      maxHeight: "287px !important",
      overflowX: "auto !important",
    },

    MuiTablePagination: {
      select: {
        paddingLeft: "8px",
        paddingRight: "20px",
        paddingTop: "10px",
        paddingBottom: "3px",
        fontSize: "10px",
      }
    }

  }
})
