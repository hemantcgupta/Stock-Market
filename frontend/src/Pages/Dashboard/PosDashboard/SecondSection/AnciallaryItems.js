import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent';
import String from '../../../../Constants/validator';
import TotalRow from '../../../../Component/TotalRow';

const apiServices = new APIServices();

class AnciallaryItems extends Component {

  constructor(props) {
    super(props);
    this.state = {
      anciallaryData: [],
      anciallaryColumn: [],
      totalData: [],
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, posDashboard } = props;
    this.url = '/pos';
    this.getPOSURL(regionId, countryId, cityId);

    self.setState({ loading: true, anciallaryData: [] })
    apiServices.getAnciallaryItems(startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, posDashboard).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var anciallaryData = result[0].ancillaryDetails;
        var totalData = result[0].totalData;
        self.setState({ anciallaryData: anciallaryData, anciallaryColumn: columnName, totalData: totalData })
      }
    });

  }

  getPOSURL(regionId, countryId, cityId, ODId) {
    let commonOD = window.localStorage.getItem('ODSelected')
    if (regionId !== 'Null') {
      this.url = `/pos?Region=${String.removeQuotes(regionId)}`
    }
    if (countryId !== 'Null') {
      this.url = `/pos?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
    }
    if (cityId !== 'Null') {
      this.url = `/pos?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&POS=${String.removeQuotes(cityId)}`
    }
    if (ODId !== 'Null') {
      this.url = `/pos?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&POS=${String.removeQuotes(cityId)}&${encodeURIComponent('O&D')}=${String.removeQuotes(ODId)}`
    }
  }

  render() {
    let ancillaryURL = `?ancillary=${true}`;
    if (this.url !== '/pos') {
      ancillaryURL = `&ancillary=${true}`
    }
    return (
      <div>
        <div className="x_title">
          <h2>Ancillary Items</h2>
          <ul className="nav navbar-right panel_toolbox">
            {/* <li><a className="collapse-link"><i className="fa fa-chevron-up"></i></a>
                </li>
                <li className="dropdown">
                  <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i className="fa fa-wrench"></i></a>
                  <ul className="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li> */}
            <li onClick={() => this.props.history.push(`${this.url}${ancillaryURL}`)}><i className="fa fa-line-chart"></i></li>
            {/* <li><a className="close-link"><i className="fa fa-close"></i></a></li> */}
          </ul>
        </div>
        <div className="x_content">
          <DataTableComponent
            rowData={this.state.anciallaryData}
            columnDefs={this.state.anciallaryColumn}
            dashboard={true}
            loading={this.state.loading}
            height={'20vh'}
          />
          <TotalRow
            rowData={this.state.totalData}
            columnDefs={this.state.anciallaryColumn}
            loading={this.state.loading}
            dashboard={true}
            reducingPadding={true}
          />
        </div>
      </div >
    )
  }
}

export default AnciallaryItems;