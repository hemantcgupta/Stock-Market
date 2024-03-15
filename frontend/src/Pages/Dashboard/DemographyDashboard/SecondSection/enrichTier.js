import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import TotalRow from '../../../../Component/TotalRow'

const apiServices = new APIServices();

class EnrichTierTable extends Component {

  constructor(props) {
    super(props);
    this.state = {
      enrichTierTable: [],
      enrichTierColumn: [],
      totalData: [],
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, regionId, countryId, cityId } = props;
    self.setState({ loading: true, enrichTierTable: [] })
    apiServices.getEnrichTiertable(startDate, endDate, regionId, countryId, cityId).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var channelwiseperformpivot = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ enrichTierTable: channelwiseperformpivot, enrichTierColumn: columnName, totalData: totalData })
      }
    });
  }

  render() {

    return (
      <div>
        <div className="x_title">
          <h2>Enrich Tier</h2>
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
            {/* <li onClick={() => this.props.history.push("/channelPerformance")}><i className="fa fa-line-chart"></i></li> */}
            {/* <li><a className="close-link"><i className="fa fa-close"></i></a></li> */}
          </ul>
        </div>
        <div className="x_content">
          <DataTableComponent
            rowData={this.state.enrichTierTable}
            columnDefs={this.state.enrichTierColumn}
            dashboard={true}
            loading={this.state.loading}
            height={'20vh'}
          />
          <TotalRow
            rowData={this.state.totalData}
            columnDefs={this.state.enrichTierColumn}
            loading={this.state.loading}
            dashboard={true}
            reducingPadding={true}
          />
        </div>
      </div>
    )
  }
}

export default EnrichTierTable;