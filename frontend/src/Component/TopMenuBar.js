import Divider from "@material-ui/core/Divider";
import IconButton from "@material-ui/core/IconButton";
import Link from "@material-ui/core/Link";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemIcon from "@material-ui/core/ListItemIcon";
import CloseIcon from "@material-ui/icons/Close";
import FiberManualRecordIcon from "@material-ui/icons/FiberManualRecord";
import LockIcon from "@material-ui/icons/Lock";
import PersonIcon from "@material-ui/icons/Person";
import VpnKeyIcon from "@material-ui/icons/VpnKey";
import React from "react";
import api from "../API/api";
import APIServices from "../API/apiservices";
import eventApi from "../API/eventApi";
import Access from "../Constants/accessValidation";
import config from "../Constants/config";
import cookieStorage from "../Constants/cookie-storage";
import { images } from "../Constants/images";
import Constants from "../Constants/validator";
import "./component.scss";
import RegionsDropDown from "./RegionsDropDown";
import RouteRegionsDropDown from "./RouteRegionsDropDown";
import Sidebar from "./Sidebar";

const apiServices = new APIServices();

class TopMenuBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      routeGroup: "",
      regionSelected: "Null",
      countrySelected: "Null",
      citySelected: "Null",
      ODSelected: "Null",
      routeSelected: "",
      menuData: [],
      startDate: "",
      endDate: "",
      gettingMonth: null,
      gettingYear: null,
      initial: "",
      toggleList: false,
      notification: false,
      alertData: [],
      isNewAlert: false,
      loading: false,
      userDetails: [],
      dashboardList: [],
      selectedDashboardPath: "",
      posFiltersSearchURL: "",
      dashboardName: '',
    };
  }

  componentWillMount() {
    const self = this;
    let user = cookieStorage.getCookie("userDetails");
    user = user ? JSON.parse(user) : "";
    let menus = cookieStorage.getCookie("menuData");
    menus = menus ? JSON.parse(menus) : "";
    let dashboardList = menus.filter((d) => d.Name === "Dashboard");
    let menusWithoutDashboard = menus.filter((d) => d.Name !== "Dashboard");
    self.setState(
      {
        userDetails: user,
        menuData: menusWithoutDashboard,
        dashboardList: dashboardList[0].submenu,
        selectedDashboardPath: window.location.pathname,
      },
      () => {
        self.getUserInitial();
        self.getAlertsForNotification(user);
      }
    );
  }

  getAlertsForNotification(userData) {
    api
      .get(`getalertnotfication`, "hideloader")
      .then((response) => {
        if (response && response.data.response.length > 0) {
          let data = response.data.response;
          const timeStamp = data[0].TimeStamp;
          this.setState({ alertData: data });
          apiServices
            .getUserPreferences("Alert_timestamp", timeStamp.toString())
            .then((result) => {
              this.setState({ isNewAlert: result });
            });
        }
      })
      .catch((err) => {
        console.log("alert notification error", err);
      });
  }

  getUserInitial() {
    const userName = this.state.userDetails.username;
    const initial = userName.charAt(0).toUpperCase();
    this.setState({ initial });
  }

  getFilterValues = ($event) => {
    let posFiltersSearchURL = Constants.getPOSFiltersSearchURL();
    let dashboardList = this.state.dashboardList.filter((d) => {
      if (d.Path.includes("forecastAccuracy")) {
        d.Path = `/forecastAccuracy${posFiltersSearchURL}`;
      }
      if (d.Path.includes("rpsPos")) {
        d.Path = `/rpsPos${posFiltersSearchURL}`;
      }
      return d;
    });
    this.setState(
      {
        regionSelected: $event.regionSelected,
        countrySelected: $event.countrySelected,
        citySelected: $event.citySelected,
        ODSelected: $event.ODSelected,
        startDate: $event.startDate,
        endDate: $event.endDate,
        gettingMonth: $event.gettingMonthA,
        gettingYear: $event.gettingYearA,
        posFiltersSearchURL: posFiltersSearchURL,
        dashboardList,
      },
      () => this.props.getPOSSelectedGeographicalDetails(this.state)
    );
  };

  getRouteFilterValues = ($event) => {
    this.setState(
      {
        routeGroup: $event.routeGroup,
        regionSelected: $event.regionSelected,
        countrySelected: $event.countrySelected,
        routeSelected: $event.routeSelected,
        startDate: $event.startDate,
        endDate: $event.endDate,
        typeofCost: $event.typeofCost,
      },
      () => this.props.getSelectedGeographicalDetails(this.state)
    );
  };

  handleDatePicker = (item) => {
    this.setState({ datePickerValue: [item.selection] });
  };

  dateSelected() {
    const datePickerValue = this.state.datePickerValue;
    const startDate = datePickerValue[0].startDate.toDateString();
    const endDate = datePickerValue[0].endDate.toDateString();
    this.setState({ showDatePicker: false, date: `${startDate} - ${endDate}` });
    this.props.getSelectedGeographicalDetails(this.state);
  }

  callDashboards = (e) => {
    this.props.history.push(`${e.target.value}`);
    let DashboardValue = e.target.value;
    this.setState({ dashboardName: DashboardValue })
  };

  _logout = () => {
    var eventData = {
      event_id: "5",
      description: "User logged out from the system",
      where_path: "/",
      page_name: "Login Page",
    };
    eventApi.sendEvent(eventData);
    cookieStorage.deleteCookie();
  };

  toggleList() {
    this.setState({ toggleList: !this.state.toggleList, notification: false });
  }

  toggleNotification() {
    this.setState({
      notification: !this.state.notification,
      toggleList: false,
      isNewAlert: false,
    });
  }

  // demographyDashboardHandler() {
  //   this.props.history.push(`/demographyDashboard`);
  // }

  renderDashboardList = () => {
    let dashboardListItems = this.state.dashboardList.map((li) => (
      <option
        value={li.Path}
        selected={li.Path === this.state.selectedDashboardPath}
      >
        {li.Name}
      </option>
    ));
    return (
      <ul className="nav navbar-nav navbar-left">
        <li
          style={{
            marginLeft: "10px",
            marginTop: "10px",
            marginBottom: "-2px",
          }}
        >
          <div className="">
            <div className="form-group dashboard-select">
              <select
                className="form-control cabinselect dashboard-dropdown"
                onChange={(e) => this.callDashboards(e)}
                id="dashboardlist"
              >
                {dashboardListItems}
              </select>
            </div>
          </div>
        </li>
      </ul>
    );
  };

  renderFilters = () => {
    return (
      <div style={{ display: "flex", alignItems: "center" }}>
        <li style={{ marginRight: "10px" }}>
          {this.props.routeDash ? (
            <RouteRegionsDropDown
              getRouteFilterValues={this.getRouteFilterValues}
              dashboardName={this.props.dashboardName}
              {...this.props}
            />
          ) : (
            <RegionsDropDown
              dashboard={true}
              demography={true}
              agentDashboard={this.props.agentDashboard}
              getFilterValues={this.getFilterValues}
              {...this.props}
            />
          )}
        </li>
      </div>
    );
  };

  renderProfileDropdown = () => {
    return (
      <div className={`profile-dropdown`}>
        <List component="nav" aria-label="main mailbox folders">
          <ListItem button>
            <ListItemIcon color="inherit">
              <PersonIcon />
            </ListItemIcon>
            <Link onClick={() => this.props.history.push("/userProfile")}>
              {"Profile"}
            </Link>
          </ListItem>
          <Divider />

          <ListItem button>
            <ListItemIcon color="inherit">
              <VpnKeyIcon />
            </ListItemIcon>
            <Link onClick={() => this.props.history.push("/changePassword")}>
              {"Change Password"}
            </Link>
          </ListItem>
          <Divider />

          <ListItem button>
            <ListItemIcon color="inherit">
              <LockIcon />
            </ListItemIcon>
            <a
              href={`${config.API_URL}/initiatelogout`}
              onClick={() => this._logout()}
            >
              {"Logout"}
            </a>
          </ListItem>
        </List>
      </div>
    );
  };

  renderNotificationDropdown = () => {
    return (
      <div className="notification-dropdown">
        <div className="notification-top">
          <div>
            <span className="count">{this.state.alertData.length}</span>New
            notifications
          </div>
          <i
            className="fa fa-close"
            aria-hidden="true"
            onClick={() => this.setState({ notification: false })}
          ></i>
        </div>
        <div className="notifications">
          {this.state.loading ? (
            <div className="ellipsis-loader">
              <img src={images.ellipsis_loader} alt="" />
            </div>
          ) : this.state.alertData.length > 0 ? (
            <div>
              {this.state.alertData.map((data) => (
                <div className="notification">
                  <div className="heading">{`POS : ${data.POS}`}</div>
                  <div className="actions">
                    {`Infare rates are`}
                    <span
                      className={`${data.IStatus === "Above" ? "green" : "red"
                        }`}
                    >{` ${data.IStatus}`}</span>
                  </div>
                </div>
              ))}
              <i
                className="bottom"
                onClick={() => this.props.history.push("/alerts")}
              >
                Show more
              </i>
            </div>
          ) : (
            <div className="ellipsis-loader">
              <h4>No new notifications to show</h4>
            </div>
          )}
        </div>
      </div>
    );
  };

  render() {
    const {
      userDetails,
      menuData,
      toggleList,
      initial,
      notification,
      isNewAlert,
    } = this.state;
    return (
      <nav className="navbar navbar-default navbar-fixed-top">
        <div className="container">
          <div className="navbar-header side-bar">
            <Sidebar
              userData={userDetails}
              menuData={menuData}
              {...this.props}
            />
            <div
              onClick={() =>
                this.props.history.push(
                  this.props.dashboardPath
                    ? this.props.dashboardPath
                    : Access.dashboardPath()
                )
              }
            >
              <img className="logo" src={images.airline_logo} alt=""></img>
            </div>
          </div>

          {this.props.dashboard ? this.renderDashboardList() : <div />}
          {/* <button style={{backgroundColor:"black"}} onClick={() => this.demographyDashboardHandler()}>
            demographyDashboard
          </button> */}
          <div id="navbar" className="navbar-collapse collapse">
            <ul className="nav navbar-nav navbar-right">
              {this.props.dashboard ? this.renderFilters() : <div />}

              {Access.accessValidation("Alert", "alerts") ? (
                <li className="notification">
                  {isNewAlert ? <div className="circle-bell"></div> : ""}
                  <i
                    className="fa fa-bell-o"
                    aria-hidden="true"
                    onClick={() => this.toggleNotification()}
                  ></i>
                </li>
              ) : (
                <li />
              )}

              <IconButton
                style={{ padding: toggleList ? "11px" : "1px" }}
                aria-label="open drawer"
                onClick={() => this.toggleList()}
                edge="start"
              >
                {toggleList ? (
                  <CloseIcon className="close-btn" />
                ) : (
                  <div className="initial-container">
                    <h3 className="initial">{initial}</h3>
                    <FiberManualRecordIcon className="circle" />
                  </div>
                )}
              </IconButton>
            </ul>

            {toggleList ? this.renderProfileDropdown() : <div />}
            {notification}
            {notification ? this.renderNotificationDropdown() : <div />}
          </div>
        </div>
      </nav>
    );
  }
}

export default TopMenuBar;
