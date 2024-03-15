import * as React from "react";
import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import { Typography } from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import { Component } from "ag-grid-community";
import Loader from "../../Component/Loader";
import TopMenuBar from "../../Component/TopMenuBar";
import { Image } from "../../Constants/images";
import "./RPSSelection.scss";
import { event } from "jquery";
import config from "../.././Constants/config";
import $ from "jquery";
import axios from "axios";
import cookieStorage from "../../Constants/cookie-storage";
import Spinners from "../../spinneranimation";
import String from "../../Constants/validator";
import Swal from "sweetalert2";

// function valuetext(value) {
//     return `${value}°C`;
// }

// const valuetext = (value) => {
//     return value
// }

// const valuetextRight = (Rvalue) => {
//     return Rvalue
// }

const API_URL = config.API_URL;

class RangeSlider extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      value: [2017, 2019],
      flownValue: [2017, 2018, 2019],
      flownValueLength: 3,
      WPGFControlvalue: [-100, 100],
      WPGFFirstControlValue: -100,
      WPGFLastControlValue: 100,
      CabinFValue: [500, 155000],
      CabinFirstFValue: 500,
      CabinLastFValue: 155000,
      CabinJValue: [250, 125000],
      CabinFirstJValue: 250,
      CabinLastJValue: 125000,
      CabinYValue: [100, 60000],
      CabinFirstYValue: 100,
      CabinLastYValue: 60000,
      CYear: new Date().getFullYear(),
      TYear: new Date().getFullYear() + 1,
      ChannelYear: new Date().getFullYear(),
      SegmentationYear: new Date().getFullYear(),
      YearsAvailable: [],
      dataAvailable: false,
      zeroInputValue: "",
      firstInputValue: 50,
      secondInputValue: 50,
      thirdInputValue: 0,
      fourthInputValue: 0,
      fifthInputValue: 50,
      sixthInputValue: 25,
      seventhInputValue: 25,
      eightthInputValue: 0,
      ninthInputValue: 0,
      tenthInputValue: "10",
      eleventhInputValue: "4",
      twelthInputValue: "6",
      thirteenthInputValue: "10",
    };
  }
  // const [value, setValue] = React.useState([2017, 2019]);
  // const [Rvalue, setRValue] = React.useState([0]);
  // const [CYear, setCYear] = React.useState('')
  // const [TYear, setTYear] = React.useState('')

  getDefaultHeader = () => {
    const token = cookieStorage.getCookie("Authorization");
    return {
      headers: {
        Authorization: token,
      },
    };
  };

  componentDidMount = () => {
    this.rpsChecks();
  };

  rpsChecks = () => {
    var regionId = window.localStorage.getItem("regionSelected");
    Swal.fire({
      title: "Processing",
      text: `Checking Data Availability`,
      icon: "info",
      confirmButtonText: "Ok",
    });
    Swal.showLoading();
    axios
      .get(
        `${API_URL}/RPSDataAvailability?regionId=${regionId}`,
        this.getDefaultHeader()
      )
      .then((respo) => {
        Swal.fire({
          title: "Completed",
          text: `Process has been Completed`,
          icon: "success",
          confirmButtonText: "Ok",
        });
        var response = respo.data.response;
        this.setState(
          {
            dataAvailable: true,
            YearsAvailable: response,
            CYear: response.CurrentYear_response[0].year,
            TYear: response.TargetYear_response[0].year,
            ChannelYear: response.channel_response[0].YearCalendar,
            SegmentationYear: response.segment_response[0].YearCalendar,
          },
          () => this.StoreValues()
        );
      })
      .catch((error) => {
        console.log(error, "error");
        this.setState({ dataAvailable: true }, () => this.StoreValues());
      });
  };

  valuetext = (value) => {
    return value;
  };

  valuetextRight = (Rvalue) => {
    return Rvalue;
  };

  handleChange = (event, newValue) => {
    let { value, flownValue } = this.state;
    console.log(newValue[0], newValue[1], "check");
    if (newValue[0] + 4 < newValue[1]) {
      alert("Cannot Select More Than 5 Years");
    } else {
      if (newValue[0] == newValue[1]) {
        console.log("1Year");
        let StoredValue = [];
        StoredValue[0] = newValue[0];
        this.setState({
          value: newValue,
          flownValue: StoredValue,
          flownValueLength: StoredValue.length,
          firstInputValue: 50,
          secondInputValue: 50,
          thirdInputValue: 0,
          fourthInputValue: 0,
          sixthInputValue: 25,
          seventhInputValue: 25,
          eightthInputValue: 0,
          ninthInputValue: 0,
        });
        console.log(flownValue.length, "length");
      } else if (newValue[0] + 1 == newValue[1]) {
        console.log("2Year");
        let StoredValue = [];
        StoredValue[0] = newValue[0];
        StoredValue[1] = newValue[1];
        this.setState({
          value: newValue,
          flownValue: StoredValue,
          flownValueLength: StoredValue.length,
          secondInputValue: 50,
          thirdInputValue: 0,
          fourthInputValue: 0,
          seventhInputValue: 25,
          eightthInputValue: 0,
          ninthInputValue: 0,
          zeroInputValue: "",
        });
        console.log(flownValue.length, "length");
      } else if (newValue[0] + 2 == newValue[1]) {
        console.log("3Year");
        let StoredValue = [];
        StoredValue[0] = newValue[0];
        StoredValue[1] = newValue[0] + 1;
        StoredValue[2] = newValue[1];
        this.setState({
          value: newValue,
          flownValue: StoredValue,
          flownValueLength: StoredValue.length,
          thirdInputValue: 0,
          fourthInputValue: 0,
          eightthInputValue: 0,
          ninthInputValue: 0,
          zeroInputValue: "",
        });
        console.log(flownValue.length, "length");
      } else if (newValue[0] + 3 == newValue[1]) {
        console.log("4Year");
        let StoredValue = [];
        StoredValue[0] = newValue[0];
        StoredValue[1] = newValue[0] + 1;
        StoredValue[2] = newValue[0] + 2;
        StoredValue[3] = newValue[1];
        this.setState({
          value: newValue,
          flownValue: StoredValue,
          flownValueLength: StoredValue.length,
          fourthInputValue: 0,
          ninthInputValue: 0,
          zeroInputValue: "",
        });
        console.log(flownValue.length, "length");
      } else if (newValue[0] + 4 == newValue[1]) {
        console.log("5Year");
        let StoredValue = [];
        StoredValue[0] = newValue[0];
        StoredValue[1] = newValue[0] + 1;
        StoredValue[2] = newValue[0] + 2;
        StoredValue[3] = newValue[0] + 3;
        StoredValue[4] = newValue[1];
        this.setState({
          value: newValue,
          flownValue: StoredValue,
          flownValueLength: StoredValue.length,
          zeroInputValue: "",
        });
        console.log(flownValue.length, "length");
      } else {
        alert("Cannot Select More Than 5 Years");
        let newValue = "";
      }
    }
    if (this.state.flownValue.length == 1) {
      this.setState({
        zeroInputValue: "Null",
        WPGFFirstControlValue: 0,
        WPGFLastControlValue: 0,
      });
    }
  };

  // handleChange = (event, newValue) => {
  //     let { value } = this.state
  //     if (newValue[0] + 1 == newValue[1]) {
  //         alert('Invalid! Minimum 3 Year Required')
  //         let UpdateValue = ''
  //     } else if (newValue[1] == value[1] + 2) {
  //         alert('Invalid! Minimum 3 Year Required')
  //         let UpdateValue = ''
  //     } else if (newValue[0] == value[0] - 2) {
  //         alert('Invalid! Minimum 3 Year Required')
  //         let UpdateValue = ''
  //     }
  //     else {
  //         let UpdateValue = newValue
  //         if (UpdateValue[1] !== value[1]) {
  //             UpdateValue[0] = UpdateValue[0] + 1
  //             UpdateValue[1] = UpdateValue[1]
  //             let StoredValue = []
  //             StoredValue[0] = UpdateValue[0]
  //             StoredValue[1] = UpdateValue[0] + 1
  //             StoredValue[2] = UpdateValue[1]
  //             this.setState({ value: UpdateValue, flownValue: StoredValue })
  //         }
  //         else if (UpdateValue[0] !== value[0]) {
  //             UpdateValue[1] = UpdateValue[1] - 1
  //             UpdateValue[0] = UpdateValue[0]
  //             let StoredValue = []
  //             StoredValue[0] = UpdateValue[0]
  //             StoredValue[1] = UpdateValue[0] + 1
  //             StoredValue[2] = UpdateValue[1]
  //             this.setState({ value: UpdateValue, flownValue: StoredValue })
  //         }
  //     }
  // };

  handleWPGFControlChange = (event, newValue) => {
    this.setState({
      WPGFControlvalue: newValue,
      WPGFFirstControlValue: newValue[0],
      WPGFLastControlValue: [1],
    });
  };

  handleCabinFChange = (event, newValue) => {
    this.setState({
      CabinFValue: newValue,
      CabinFirstFValue: newValue[0],
      CabinLastFValue: newValue[1],
    });
  };

  handleCabinJChange = (event, newValue) => {
    this.setState({
      CabinJValue: newValue,
      CabinFirstJValue: newValue[0],
      CabinLastJValue: newValue[1],
    });
  };

  handleCabinYChange = (event, newValue) => {
    this.setState({
      CabinYValue: newValue,
      CabinFirstYValue: newValue[0],
      CabinLastYValue: newValue[1],
    });
  };

  CurrentYear = (e) => {
    e.persist();
    this.setState({ CYear: e.target.value });
  };

  TargetYear = (e) => {
    let TargetYearVal = e.target.value;
    this.setState({ TYear: TargetYearVal });
  };

  ChannelYear = (e) => {
    let ChannelYearVal = e.target.value;
    this.setState({ ChannelYear: ChannelYearVal });
  };

  SegmentationYear = (e) => {
    let SegmentationYearVal = e.target.value;
    this.setState({ SegmentationYear: SegmentationYearVal });
  };

  firstInput = (e) => {
    e.persist();
    this.setState({ firstInputValue: e.target.value });
  };

  secondInput = (e) => {
    e.persist();
    this.setState({ secondInputValue: e.target.value });
  };

  thirdInut = (e) => {
    e.persist();
    this.setState({ thirdInputValue: e.target.value });
  };

  fourthInput = (e) => {
    e.persist();
    this.setState({ fourthInputValue: e.target.value });
  };

  fifthInput = (e) => {
    e.persist();
    this.setState({ fifthInputValue: e.target.value });
  };

  sixthInput = (e) => {
    e.persist();
    this.setState({ sixthInputValue: e.target.value });
  };

  seventhInput = (e) => {
    e.persist();
    this.setState({ seventhInputValue: e.target.value });
  };

  eightthInput = (e) => {
    e.persist();
    this.setState({ eightthInputValue: e.target.value });
  };

  ninthInput = (e) => {
    e.persist();
    this.setState({ ninthInputValue: e.target.value });
  };

  tenthInput = (e) => {
    e.persist();
    this.setState({ tenthInputValue: e.target.value });
  };

  eleventhInput = (e) => {
    e.persist();
    this.setState({ eleventhInputValue: e.target.value });
  };

  twelthInput = (e) => {
    e.persist();
    this.setState({ twelthInputValue: e.target.value });
  };

  thirteenthInput = (e) => {
    e.persist();
    this.setState({ thirteenthInputValue: e.target.value });
  };

  StoreValues = () => {
    let {
      flownValue,
      flownValueLength,
      WPGFFirstControlValue,
      WPGFLastControlValue,
      CabinFirstFValue,
      CabinLastFValue,
      CabinFirstJValue,
      CabinLastJValue,
      CabinFirstYValue,
      CabinLastYValue,
      CYear,
      TYear,
      zeroInputValue,
      firstInputValue,
      secondInputValue,
      thirdInputValue,
      fourthInputValue,
      fifthInputValue,
      sixthInputValue,
      seventhInputValue,
      eightthInputValue,
      ninthInputValue,
      tenthInputValue,
      eleventhInputValue,
      twelthInputValue,
      thirteenthInputValue,
      ChannelYear,
      SegmentationYear,
    } = this.state;

    let StoredValues = {
      FlownValue: flownValue,
      FlownValueLength: flownValueLength,
      WPGFFirstValue: WPGFFirstControlValue,
      WPGFLastValue: WPGFLastControlValue,
      CabinFirstFValue: CabinFirstFValue,
      CabinLastFValue: CabinLastFValue,
      CabinFirstJValue: CabinFirstJValue,
      CabinLastJValue: CabinLastJValue,
      CabinFirstYValue: CabinFirstYValue,
      CabinLastYValue: CabinLastYValue,
      CabinLastYValue: CabinLastYValue,
      CYearValue: CYear,
      TYearValue: TYear,
      ZeroInputValue: zeroInputValue,
      FirstInputValue: firstInputValue,
      SecondInputValue: secondInputValue,
      ThirdInputValue: thirdInputValue,
      FourthInputValue: fourthInputValue,
      FifthInputValue: fifthInputValue,
      SixthInputValue: sixthInputValue,
      SeventhInputValue: seventhInputValue,
      EightthInputValue: eightthInputValue,
      NinthInputValue: ninthInputValue,
      TenthInputValue: tenthInputValue,
      EleventhInputValue: eleventhInputValue,
      TwelthInputValue: twelthInputValue,
      ThirteenthInputValue: thirteenthInputValue,
      ChannelYearValue: ChannelYear,
      SegmentationYearValue: SegmentationYear,
    };
    window.localStorage.setItem(
      "RPSAdminScreenValues",
      JSON.stringify(StoredValues)
    );
  };

  SubmitValues = () => {
    let {
      flownValue,
      flownValueLength,
      WPGFFirstControlValue,
      WPGFLastControlValue,
      CabinFirstFValue,
      CabinLastFValue,
      CabinFirstJValue,
      CabinLastJValue,
      CabinFirstYValue,
      CabinLastYValue,
      CYear,
      TYear,
      zeroInputValue,
      firstInputValue,
      secondInputValue,
      thirdInputValue,
      fourthInputValue,
      fifthInputValue,
      sixthInputValue,
      seventhInputValue,
      eightthInputValue,
      ninthInputValue,
      tenthInputValue,
      eleventhInputValue,
      twelthInputValue,
      thirteenthInputValue,
      ChannelYear,
      SegmentationYear,
    } = this.state;
    let totalValue =
      parseInt(firstInputValue) +
      parseInt(secondInputValue) +
      parseInt(thirdInputValue) +
      parseInt(fourthInputValue);
    let totalValueLower =
      parseInt(fifthInputValue) +
      parseInt(sixthInputValue) +
      parseInt(seventhInputValue) +
      parseInt(eightthInputValue) +
      parseInt(ninthInputValue);
    console.log(totalValue, totalValueLower, zeroInputValue, "totalvalue");
    if (
      (totalValue == 100 && totalValueLower == 100) ||
      (zeroInputValue == "Null" && totalValueLower == 100)
    ) {
      let StoredValues = {
        FlownValue: flownValue,
        FlownValueLength: flownValueLength,
        WPGFFirstValue: WPGFFirstControlValue,
        WPGFLastValue: WPGFLastControlValue,
        CabinFirstFValue: CabinFirstFValue,
        CabinLastFValue: CabinLastFValue,
        CabinFirstJValue: CabinFirstJValue,
        CabinLastJValue: CabinLastJValue,
        CabinFirstYValue: CabinFirstYValue,
        CabinLastYValue: CabinLastYValue,
        CabinLastYValue: CabinLastYValue,
        CYearValue: CYear,
        TYearValue: TYear,
        ZeroInputValue: zeroInputValue,
        FirstInputValue: firstInputValue,
        SecondInputValue: secondInputValue,
        ThirdInputValue: thirdInputValue,
        FourthInputValue: fourthInputValue,
        FifthInputValue: fifthInputValue,
        SixthInputValue: sixthInputValue,
        SeventhInputValue: seventhInputValue,
        EightthInputValue: eightthInputValue,
        NinthInputValue: ninthInputValue,
        TenthInputValue: tenthInputValue,
        EleventhInputValue: eleventhInputValue,
        TwelthInputValue: twelthInputValue,
        ThirteenthInputValue: thirteenthInputValue,
        ChannelYearValue: ChannelYear,
        SegmentationYearValue: SegmentationYear,
      };
      window.localStorage.setItem(
        "RPSAdminScreenValues",
        JSON.stringify(StoredValues)
      );
      Swal.fire({
        title: "Processing",
        text: `Demand Estimation process has been started, once the process complete we will notify you.`,
        icon: "info",
        confirmButtonText: "Ok",
      });
      Swal.showLoading();
      axios
        .get(
          `${API_URL}/RPSDECalculation?flownValue=${flownValue}&flownValueLength=${flownValueLength}&WPGFControlInitialValue=${WPGFFirstControlValue}&WPGFControlFinalValue=${WPGFLastControlValue}&CabinFValueInitial=${CabinFirstFValue}&CabinFValueFinal=${CabinLastFValue}&CabinJValueInitial=${CabinFirstJValue}&CabinJValueFinal=${CabinLastJValue}&CabinYValueInitial=${CabinFirstYValue}&CabinYValueFinal=${CabinLastYValue}&CurrentYear=${CYear}&TargetYear=${TYear}&WPGFfirstInput=${firstInputValue}&WPGFsecondInput=${secondInputValue}&WPGFthirdInput=${thirdInputValue}&WPGFfourthInput=${fourthInputValue}&WMSFfifthInput=${fifthInputValue}&WMSFsixthInput=${sixthInputValue}&WMSFseventhInput=${seventhInputValue}&WMSFeightthInput=${eightthInputValue}&WMSFninthInput=${ninthInputValue}&tenthInput=${tenthInputValue}&eleventhInput=${eleventhInputValue}&twelthInput=${twelthInputValue}&thirteenthInput=${thirteenthInputValue}&channelYear=${ChannelYear}&segmentationYear=${SegmentationYear}`,
          this.getDefaultHeader()
        )
        .then((response) => {
          Swal.fire({
            title: "Completed",
            text: `Process has been Completed`,
            icon: "success",
            confirmButtonText: "Ok",
          });
          this.props.history.push("/rpsPos");
        })
        .catch((error) => {
          Swal.fire({
            title: "Failed",
            text: `Process has been failed Please Retry else Contact Revemax`,
            icon: "error",
            confirmButtonText: "Ok",
          });
        });
    } else {
      alert("Growth Parameters Weightage Must Be Equal to 100 ! ");
    }
  };

  showLoader = () => {
    $("#loaderImage").addClass("loader-visible");
  };

  // hideLoader = () => {
  //     $("#loaderImage").removeClass("loader-visible")
  //     $(".x_panel").addClass("opacity-fade");
  //     $(".top-buttons").addClass("opacity-fade");
  // }

  render() {
    let {
      value,
      WPGFControlvalue,
      dataAvailable,
      CabinFValue,
      CabinJValue,
      CabinYValue,
      flownValue,
      YearsAvailable,
    } = this.state;
    let {
      firstInputValue,
      secondInputValue,
      thirdInputValue,
      fourthInputValue,
      flownValueLength,
      fifthInputValue,
      sixthInputValue,
      seventhInputValue,
      eightthInputValue,
      ninthInputValue,
      zeroInputValue,
    } = this.state;
    let totalValue =
      parseInt(firstInputValue) +
      parseInt(secondInputValue) +
      parseInt(thirdInputValue) +
      parseInt(fourthInputValue);
    console.log(
      zeroInputValue,
      firstInputValue,
      secondInputValue,
      thirdInputValue,
      fourthInputValue,
      totalValue,
      flownValueLength,
      fifthInputValue,
      sixthInputValue,
      seventhInputValue,
      eightthInputValue,
      ninthInputValue,
      "totalvalue"
    );
    return (
      <div className="pos-details">
        <TopMenuBar dashboardPath={"/posDashboard"} {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: "0px" }}>
              <div className="col-md-7 col-sm-7 col-xs-7 toggle1  rpsSelectionHeading ">
                <h3>RPS Selection Page</h3>
              </div>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div
              className="x_panel MainPageBox"
              style={{
                marginTop: "10px",
                height: "calc(100vh - 130px)",
                display: "flex",
                flexWrap: "wrap",
              }}
            >
              {dataAvailable ? (
                <div className="upperBoxes">
                  <div className="OuterContainer leftBox">
                    <h2>Data Loading </h2>
                    <div>
                      <div className="Boxes1">
                        <div style={{ marginLeft: "1%" }}>
                          <Typography
                            style={{ marginBottom: "6%", fontSize: "2vh" }}
                          >
                            Flown Data
                          </Typography>
                          <Box
                            sx={{
                              width: "100%",
                              display: "flex",
                              marginLeft: "4%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                              "& .MuiSlider-track": {
                                height: 3,
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={
                                YearsAvailable.flown_response[0].yearCalendar
                              }
                              max={
                                YearsAvailable.flown_response[
                                  YearsAvailable.flown_response.length - 1
                                ].yearCalendar
                              }
                              value={value}
                              onChange={(event, e) =>
                                this.handleChange(event, e)
                              }
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetext}
                            />
                          </Box>
                        </div>
                      </div>
                      <div className="Boxes1">
                        <div style={{ marginLeft: "1%" }}>
                          <Typography style={{ marginBottom: "6%" }}>
                            <p className="MIDTReal">
                              MIDT
                              <p className="MIDT">
                                Marketing Information Data Tapes
                              </p>
                            </p>
                          </Typography>
                          <Box
                            sx={{
                              width: "100%",
                              marginLeft: "4%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={YearsAvailable.MIDT_response[0].yearCalendar}
                              max={
                                YearsAvailable.MIDT_response[
                                  YearsAvailable.MIDT_response.length - 1
                                ].yearCalendar
                              }
                              value={value}
                              onChange={this.handleChange}
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetext}
                            />
                          </Box>
                        </div>
                      </div>
                    </div>

                    <p style={{ marginLeft: "3%" }}>Capacity Data </p>
                    <div className="Boxes2">
                      <div style={{ marginRight: "8%" }}>
                        <h4>Current Year: </h4>
                        <select
                          className="form-control cabinselect currency-dropdown"
                          onChange={(e) => this.CurrentYear(e)}
                        >
                          {YearsAvailable.CurrentYear_response.map((Cyears) => (
                            <option value={Cyears.year}>{Cyears.year}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <h4>Target Year: </h4>
                        <select
                          className="form-control cabinselect currency-dropdown"
                          onChange={(e) => this.TargetYear(e)}
                        >
                          {YearsAvailable.TargetYear_response.map((Cyears) => (
                            <option value={Cyears.year}>{Cyears.year}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                  <div className="rightBox">
                    <h2>Growth Parameters</h2>
                    <div className="Box3-1 wpgf-style">
                      <p className="WPGFReal">
                        WPGF
                        <p className="WPGF">Weighted Passenger Growth Factor</p>
                      </p>

                      {/* <input type='text' placeholder='Enter here'></input> */}
                      <div
                        style={{
                          display: "flex",
                          marginTop: "-10px",
                          marginLeft: "40px",
                        }}
                      >
                        {flownValue.length == 1 ? (
                          <div style={{ marginLeft: "5%" }}>
                            <div
                              style={{
                                display: "flex",
                                borderColor: "#2e303f",
                              }}
                            >
                              <input
                                type="text"
                                className="form-control"
                                value="No Selection"
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "60%",
                                  height: "4vh",
                                }}
                              />
                            </div>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 2 ||
                        flownValue.length == 3 ||
                        flownValue.length == 4 ||
                        flownValue.length == 5 ? (
                          <div style={{ marginLeft: "5%" }}>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={50}
                                onChange={(e) => this.firstInput(e)}
                                maxLength="3"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1]}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 3 ||
                        flownValue.length == 4 ||
                        flownValue.length == 5 ? (
                          <div>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={50}
                                onChange={(e) => this.secondInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 1}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 4 || flownValue.length == 5 ? (
                          <div>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={0}
                                onChange={(e) => this.thirdInut(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 2}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 5 ? (
                          <div style={{ marginLeft: "5%" }}>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={0}
                                onChange={(e) => this.fourthInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 3}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                      </div>
                    </div>
                    <div className="Box3-1 wpgf-style bordercolor">
                      <p className="WPGFReal " style={{ marginTop: "-7px" }}>
                        WMSF
                        <p className="WPGF">Weighted Market Share Factor</p>
                      </p>
                      <div
                        style={{
                          display: "flex",
                          marginTop: "-10px",
                          marginLeft: "40px",
                        }}
                      >
                        {flownValue.length == 2 ||
                        flownValue.length == 1 ||
                        flownValue.length == 3 ||
                        flownValue.length == 4 ||
                        flownValue.length == 5 ? (
                          <div style={{ marginLeft: "3%" }}>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={50}
                                onChange={(e) => this.fifthInput(e)}
                                maxLength="3"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1]}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 2 ||
                        flownValue.length == 3 ||
                        flownValue.length == 4 ||
                        flownValue.length == 5 ? (
                          <div>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={25}
                                onChange={(e) => this.sixthInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                  marginLeft: "-2px",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 1}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 3 ||
                        flownValue.length == 4 ||
                        flownValue.length == 5 ? (
                          <div>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={25}
                                onChange={(e) => this.seventhInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 2}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 4 || flownValue.length == 5 ? (
                          <div style={{ marginLeft: "5%" }}>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={0}
                                onChange={(e) => this.eightthInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 3}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                        {flownValue.length == 5 ? (
                          <div style={{ marginLeft: "5%" }}>
                            <div style={{ display: "flex" }}>
                              <input
                                type="text"
                                className="form-control"
                                defaultValue={0}
                                onChange={(e) => this.ninthInput(e)}
                                maxLength="2"
                                style={{
                                  backgroundColor: "#2e303f",
                                  width: "26%",
                                  height: "4vh",
                                }}
                              />
                              <span className="percentSuffix"> % </span>
                            </div>
                            <p style={{ fontSize: "1.3rem", marginLeft: "8%" }}>
                              {value[1] - 4}
                            </p>
                          </div>
                        ) : (
                          ""
                        )}
                      </div>
                    </div>
                    <div className="Box3" style={{ marginTop: "-15px" }}>
                      <p className="WPGFReal">
                        TMS :<p className="WPGF">Target Market Share</p>
                      </p>
                      {/* <img src={Image.arrow_up} /> */}
                      <div style={{ display: "flex" }}>
                        {/* {ArrowUpwardIcon} */}
                        <p className="WPGFReal">
                          WMSF
                          <p className="WPGF">Weighted Market Share Factor</p>
                        </p>
                        <ArrowUpwardIcon fontSize="large" color="success" />
                        <div style={{ marginLeft: "4%" }}>
                          <div style={{ display: "flex" }}>
                            <input
                              type="text"
                              defaultValue={10}
                              className="form-control"
                              onChange={(e) => this.tenthInput(e)}
                              maxLength="2"
                              style={{
                                backgroundColor: "#2e303f",
                                width: "30%",
                                height: "4vh",
                                color: "white",
                              }}
                            />
                            <span className="percentSuffix"> % </span>
                          </div>
                          <p style={{ fontSize: "1.3rem", marginLeft: "10%" }}>
                            0%
                          </p>
                        </div>
                        <div>
                          <div style={{ display: "flex" }}>
                            <input
                              type="text"
                              defaultValue="4"
                              className="form-control"
                              onChange={(e) => this.eleventhInput(e)}
                              maxLength="2"
                              style={{
                                backgroundColor: "#2e303f",
                                width: "30%",
                                height: "4vh",
                              }}
                            />
                            <span className="percentSuffix"> % </span>
                          </div>
                          <p style={{ fontSize: "1.3rem", marginLeft: "4%" }}>
                            1-15%
                          </p>
                        </div>
                        <div>
                          <div style={{ display: "flex" }}>
                            <input
                              type="text"
                              defaultValue="6"
                              className="form-control"
                              onChange={(e) => this.twelthInput(e)}
                              maxLength="2"
                              style={{
                                backgroundColor: "#2e303f",
                                width: "30%",
                                height: "4vh",
                              }}
                            />
                            <span className="percentSuffix"> % </span>
                          </div>
                          <p style={{ fontSize: "1.3rem", marginLeft: "4%" }}>
                            16-30%
                          </p>
                        </div>
                        <div>
                          <div style={{ display: "flex" }}>
                            <input
                              type="text"
                              defaultValue="10"
                              className="form-control"
                              onChange={(e) => this.thirteenthInput(e)}
                              maxLength="2"
                              style={{
                                backgroundColor: "#2e303f",
                                width: "30%",
                                height: "4vh",
                              }}
                            />
                            <span className="percentSuffix"> % </span>
                          </div>
                          <p style={{ fontSize: "1.3rem", marginLeft: "0%" }}>
                            above 30%
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="Box3-2" style={{ marginTop: "5px" }}>
                      {flownValue.length !== 1 ? (
                        <div
                          style={{
                            marginLeft: "3%",
                            display: "flex",
                            flexDirection: "row",
                            justifyContent: "flex-start",
                            alignItems: "center",
                            marginTop: "35px",
                          }}
                        >
                          <Typography
                            style={{ marginBottom: "0%", fontSize: "2vh" }}
                          >
                            <p className="WPGFReal">
                              WPGF Control
                              <p className="WPGF">
                                Weighted Passenger Growth Factor Control
                              </p>
                            </p>
                          </Typography>
                          <Box
                            sx={{
                              width: "50%",
                              marginLeft: "10%",
                              marginTop: "-2.5%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={-100}
                              max={100}
                              value={WPGFControlvalue}
                              onChange={this.handleWPGFControlChange}
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetextRight}
                            />
                          </Box>
                        </div>
                      ) : (
                        ""
                      )}
                    </div>

                    <div className="Box3-3">
                      <p
                        style={{
                          fontSize: "2rem",
                          marginLeft: "3%",
                          marginTop: "0%",
                        }}
                      >
                        Outlier Control :
                      </p>
                      <div
                        className="Box5"
                        style={{ display: "flex", marginTop: "30px" }}
                      >
                        <div style={{ marginLeft: "12%" }}>
                          <Typography
                            style={{
                              marginBottom: "30%",
                              marginTop: "-22%",
                              fontSize: "2vh",
                            }}
                          >
                            Cabin F
                          </Typography>
                          <Box
                            sx={{
                              width: "250%",
                              marginLeft: "1%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={500}
                              max={155000}
                              value={CabinFValue}
                              onChange={this.handleCabinFChange}
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetextRight}
                            />
                          </Box>
                        </div>
                        <div style={{ marginLeft: "20%" }}>
                          <Typography
                            style={{
                              marginBottom: "30%",
                              marginTop: "-22%",
                              fontSize: "2vh",
                            }}
                          >
                            Cabin J
                          </Typography>
                          <Box
                            sx={{
                              width: "250%",
                              marginLeft: "1%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={250}
                              max={125000}
                              value={CabinJValue}
                              onChange={this.handleCabinJChange}
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetextRight}
                            />
                          </Box>
                        </div>
                        <div style={{ marginLeft: "20%" }}>
                          <Typography
                            style={{
                              marginBottom: "30%",
                              marginTop: "-22%",
                              fontSize: "2vh",
                            }}
                          >
                            Cabin Y
                          </Typography>
                          <Box
                            sx={{
                              width: "250%",
                              marginLeft: "1%",
                              "& .MuiSlider-thumb": {
                                height: "12px",
                                width: "10px",
                              },
                            }}
                          >
                            <Slider
                              getAriaLabel={() => "Temperature range"}
                              min={100}
                              max={60000}
                              value={CabinYValue}
                              onChange={this.handleCabinYChange}
                              valueLabelDisplay="on"
                              getAriaValueText={this.valuetextRight}
                            />
                          </Box>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <Spinners />
              )}
              {dataAvailable ? (
                <div className="lowerBoxes">
                  <div className="lowerLeftBox">
                    <h2>Channel Proration Year </h2>
                    <div className="lowerInsiderBox">
                      <h4 style={{ fontSize: "2rem !important" }}>Year: </h4>
                      <select
                        className="form-control cabinselect currency-dropdown"
                        onChange={(e) => this.ChannelYear(e)}
                      >
                        {YearsAvailable.channel_response.map((Cyears) => (
                          <option value={Cyears.YearCalendar}>
                            {Cyears.YearCalendar}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="lowerRightBox">
                    <h2>Segmentation Proration Year </h2>
                    <div className="lowerInsiderBox">
                      <h4>Year: </h4>
                      <select
                        className="form-control cabinselect currency-dropdown"
                        onChange={(e) => this.SegmentationYear(e)}
                      >
                        {YearsAvailable.segment_response.map((Cyears) => (
                          <option value={Cyears.YearCalendar}>
                            {Cyears.YearCalendar}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              ) : (
                <Spinners />
              )}
              <button className="submitBtn" onClick={this.SubmitValues}>
                {" "}
                Submit{" "}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default RangeSlider;
